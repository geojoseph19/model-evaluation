"""ALD API client — payload construction, request handling, response parsing."""

import asyncio
import logging
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["bn", "gu", "kn", "ml", "mr", "ta", "te"]


class ALDClient:
    def __init__(
        self,
        api_url: str,
        use_dummy: bool = True,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
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
        return {
            "detected_language": str(raw.get("detected_language", "unknown")).lower(),
            "confidence": float(raw.get("confidence", 0.0)),
            "all_scores": raw.get("all_scores", {}),
        }

    async def _real_detect(self, audio_path: Path, filename: str) -> dict:
        import aiohttp
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
                if attempt == self.retry_attempts - 1:
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
            detected = random.choice([l for l in SUPPORTED_LANGUAGES if l != ground_truth])
            confidence = random.uniform(0.35, 0.70)

        scores = {l: round(random.uniform(0.01, 0.12), 4) for l in SUPPORTED_LANGUAGES}
        scores[detected] = round(confidence, 4)
        total = sum(scores.values())
        scores = {k: round(v / total, 4) for k, v in scores.items()}

        return {"detected_language": detected, "confidence": round(confidence, 4), "all_scores": scores}
