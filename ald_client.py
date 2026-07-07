"""ALD API client — payload construction, request handling, response parsing."""

import asyncio
import importlib.util
import logging
import os
import random
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["bn", "gu", "kn", "ml", "mr", "ta", "te"]

# uvector WSSL model uses 3-letter codes; map to ISO 639-1 used by this eval framework
_LANG3_TO_ISO2 = {
    "asm": "as", "ben": "bn", "eng": "en", "guj": "gu", "hin": "hi",
    "kan": "kn", "mal": "ml", "mar": "mr", "odi": "or", "pun": "pa",
    "tam": "ta", "tel": "te",
}
_ID2LANG3 = {
    0: "asm", 1: "ben", 2: "eng", 3: "guj", 4: "hin", 5: "kan",
    6: "mal", 7: "mar", 8: "odi", 9: "pun", 10: "tam", 11: "tel",
}


class ALDClient:
    def __init__(
        self,
        api_url: str,
        use_dummy: bool = True,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        if retry_attempts < 1:
            raise ValueError(f"retry_attempts must be >= 1, got {retry_attempts}")
        self.api_url = api_url
        self.use_dummy = use_dummy
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._session = None

    async def __aenter__(self):
        if not self.use_dummy:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                raise RuntimeError("aiohttp required for real API: pip install aiohttp")
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    async def detect(self, audio_path: Path, filename: str) -> dict:
        if self.use_dummy:
            return await self._dummy_detect(filename)
        return await self._real_detect(audio_path, filename)

    def _build_payload(self, audio_bytes: bytes, filename: str):
        """Construct the multipart form payload. Edit here to add/rename fields."""
        import aiohttp
        form = aiohttp.FormData()
        form.add_field("audio", audio_bytes, filename=filename, content_type="audio/wav")
        return form

    def _parse_response(self, raw: dict) -> dict:
        """Extract fields from API response. Edit here when API schema changes."""
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "detected_language": str(raw.get("detected_language", "unknown")).lower(),
            "confidence": confidence,
            "all_scores": raw.get("all_scores", {}),
        }

    async def _real_detect(self, audio_path: Path, filename: str) -> dict:
        import aiohttp
        if self._session is None:
            raise RuntimeError("ALDClient must be used as an async context manager")
        audio_bytes = audio_path.read_bytes()

        for attempt in range(self.retry_attempts):
            try:
                form = self._build_payload(audio_bytes, filename)
                async with self._session.post(
                    self.api_url,
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    resp.raise_for_status()
                    return self._parse_response(await resp.json())
            except Exception as exc:
                is_client_error = (
                    hasattr(exc, "status") and isinstance(exc.status, int) and exc.status < 500
                )
                if attempt == self.retry_attempts - 1 or is_client_error:
                    raise
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(f"Retry {attempt + 1} for {filename}: {exc} (wait {wait:.1f}s)")
                await asyncio.sleep(wait)

    async def _dummy_detect(self, filename: str) -> dict:
        await asyncio.sleep(random.uniform(0.04, 0.25))

        parts = Path(filename).stem.rsplit("_", 1)
        duration = parts[-1] if len(parts) > 1 else ""
        lang_prefix = Path(filename).stem.split("_")[0]
        ground_truth = lang_prefix if lang_prefix in SUPPORTED_LANGUAGES else random.choice(SUPPORTED_LANGUAGES)

        accuracy = {"1s": 0.72, "2s": 0.81, "3s": 0.88, "5s": 0.93}.get(duration, 0.85)
        if random.random() < accuracy:
            detected = ground_truth
            confidence = random.uniform(0.75, 0.97)
        else:
            detected = random.choice([lang for lang in SUPPORTED_LANGUAGES if lang != ground_truth])
            confidence = random.uniform(0.35, 0.70)

        scores = {l: round(random.uniform(0.01, 0.12), 4) for l in SUPPORTED_LANGUAGES}
        scores[detected] = round(confidence, 4)
        total = sum(scores.values())
        scores = {k: round(v / total, 4) for k, v in scores.items()}
        confidence = scores[detected]  # use normalized value so confidence == all_scores[detected]

        return {"detected_language": detected, "confidence": confidence, "all_scores": scores}


class LocalModelClient:
    """Runs the uvector WSSL model in-process. Loads weights once on __aenter__."""

    def __init__(self, model_dir: str):
        self._model_dir = str(model_dir)
        self._evaluater = None
        self._net = None
        self._mod = None
        self._device = None
        self._use_fp16 = False
        self._lock = threading.Lock()  # PyTorch/wav2vec not thread-safe; serialize inference

    async def __aenter__(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load)
        return self

    async def __aexit__(self, *args):
        pass

    def _load(self):
        import sys
        import torch
        sys.path.insert(0, self._model_dir)

        from ccc_wav2vec_extractor import HiddenFeatureExtractor

        spec = importlib.util.spec_from_file_location(
            "demo_uvector_wssl",
            os.path.join(self._model_dir, "demo_uvector_wssl.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._mod = mod

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._use_fp16 = device.type == "cuda"

        if device.type == "cpu":
            torch.set_num_threads(os.cpu_count() or 1)

        model1 = mod.LSTMNet()
        model2 = mod.LSTMNet()
        net = mod.MSA_DAT_Net(model1, model2)
        weights = os.path.join(self._model_dir, "model", "ZWSSL_train_SpringData_13June2024_e3.pth")
        net.load_state_dict(torch.load(weights, map_location=device), strict=False)
        net.to(device)
        if self._use_fp16:
            net.half()
        net.eval()
        if hasattr(torch, "compile"):
            try:
                net = torch.compile(net)
            except Exception:
                pass
        self._net = net

        self._evaluater = HiddenFeatureExtractor()
        logger.info(f"LocalModelClient: loaded model from {self._model_dir} (device={device}, fp16={self._use_fp16})")

    def _detect_sync(self, audio_path: Path) -> dict:
        import numpy as np
        import torch
        from torch.autograd import Variable

        with self._lock:
            return self._detect_sync_locked(audio_path)

    def _detect_sync_locked(self, audio_path: Path) -> dict:
        import numpy as np
        import torch
        from torch.autograd import Variable

        file_names, speech_list = self._evaluater.preprocess_audio([str(audio_path)])
        if not speech_list or len(speech_list[0]) <= 16400:
            logger.warning(f"Audio too short to classify: {audio_path.name}")
            return {"detected_language": "unknown", "confidence": 0.0, "all_scores": {}}

        hidden_features = self._evaluater.hiddenFeatures([speech_list[0]])
        X1, X2 = self._mod.lstm_data(hidden_features[0])
        X1 = np.swapaxes(X1, 0, 1)
        X2 = np.swapaxes(X2, 0, 1)
        dtype = torch.float16 if self._use_fp16 else torch.float32
        x1 = Variable(X1, requires_grad=False).to(self._device, dtype=dtype)
        x2 = Variable(X2, requires_grad=False).to(self._device, dtype=dtype)

        with torch.no_grad():
            o1, _, _, _ = self._net.forward(x1, x2)

        raw = o1.detach().cpu().float().numpy()[0]
        probs = np.exp(raw) / np.sum(np.exp(raw))
        lang_idx = int(np.argmax(probs))

        lang3 = _ID2LANG3[lang_idx]
        detected = _LANG3_TO_ISO2.get(lang3, lang3)
        confidence = round(float(probs[lang_idx]), 4)
        all_scores = {
            _LANG3_TO_ISO2.get(_ID2LANG3[i], _ID2LANG3[i]): round(float(p), 4)
            for i, p in enumerate(probs)
        }
        return {"detected_language": detected, "confidence": confidence, "all_scores": all_scores}

    async def detect(self, audio_path: Path, filename: str) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._detect_sync, audio_path)
