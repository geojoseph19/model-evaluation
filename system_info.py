"""Collect hardware, OS, runtime, and process resource metadata for evaluation runs.

All functions are safe to call on any platform. Missing tools, optional
dependencies (psutil), or unsupported OS paths are silently omitted — the
caller always receives a plain dict with whatever could be determined.

Public API
----------
collect_device_info(input_dir)  -> dict   # hardware + OS + GPU + env vars
collect_runtime_info()          -> dict   # Python, packages, CUDA, env hash
take_pre_run_snapshot()         -> dict   # available memory/GPU before eval starts
collect_peak_usage()            -> dict   # OS-tracked peak RSS after eval ends
"""

import os
from pathlib import Path
from typing import Optional


def collect_device_info(input_dir: Path) -> dict:
    """Return hardware and OS details for the current machine."""
    import platform as _platform
    import re
    import subprocess

    system = _platform.system()
    info: dict = {
        "os": system,
        "os_release": _platform.release(),
        "os_version": _platform.version(),
        "machine": _platform.machine(),
    }

    # CPU model — platform.processor() returns bare arch string on Linux
    cpu = ""
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
        elif system == "Darwin":
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL, text=True, timeout=5,
            ).strip()
    except Exception:
        pass
    if not cpu:
        cpu = _platform.processor()
    if cpu:
        info["cpu"] = cpu

    # Device model name
    try:
        if system == "Linux":
            with open("/sys/devices/virtual/dmi/id/product_name", encoding="utf-8") as f:
                model = f.read().strip()
            if model and model not in ("", "To Be Filled By O.E.M."):
                info["device_model"] = model
        elif system == "Darwin":
            model = subprocess.check_output(
                ["sysctl", "-n", "hw.model"],
                stderr=subprocess.DEVNULL, text=True, timeout=5,
            ).strip()
            if model:
                info["device_model"] = model
        elif system == "Windows":
            out = subprocess.check_output(
                ["wmic", "computersystem", "get", "model", "/value"],
                stderr=subprocess.DEVNULL, text=True, timeout=5,
            ).strip()
            for line in out.splitlines():
                if line.lower().startswith("model="):
                    val = line.split("=", 1)[1].strip()
                    if val:
                        info["device_model"] = val
                    break
    except Exception:
        pass

    # CPU cores, frequency, memory, storage via psutil
    try:
        import psutil

        info["cpu_logical_cores"] = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
        if physical:
            info["cpu_physical_cores"] = physical

        try:
            freq = psutil.cpu_freq()
            if freq:
                mhz = freq.max or freq.current
                if mhz:
                    info["cpu_freq_max_mhz"] = round(mhz, 1)
        except Exception:
            pass

        mem = psutil.virtual_memory()
        info["memory_total_gb"] = round(mem.total / (1024 ** 3), 2)
        info["memory_available_gb"] = round(mem.available / (1024 ** 3), 2)

        # Measure the disk that actually holds input_dir, not always /
        try:
            disk = psutil.disk_usage(str(input_dir.resolve()))
        except Exception:
            disk = psutil.disk_usage("C:\\" if system == "Windows" else "/")
        info["storage_total_gb"] = round(disk.total / (1024 ** 3), 2)
        info["storage_free_gb"] = round(disk.free / (1024 ** 3), 2)

        # Storage type — Linux only via sysfs
        if system == "Linux":
            try:
                partitions = psutil.disk_partitions(all=False)
                input_str = str(input_dir.resolve())
                best = max(
                    (p for p in partitions if input_str.startswith(p.mountpoint)),
                    key=lambda p: len(p.mountpoint),
                    default=None,
                )
                if best:
                    dev_name = os.path.basename(best.device)
                    # Strip partition suffix without removing the disk's own trailing number.
                    # NVMe/MMC/nbd devices use a "p<N>" partition suffix (nvme0n1p1→nvme0n1,
                    # mmcblk0p1→mmcblk0).  SCSI/SATA/virtio devices suffix partitions with a
                    # plain digit (sda1→sda, vda2→vda).  Loop devices are virtual — no strip.
                    if any(dev_name.startswith(pfx) for pfx in ("nvme", "mmcblk", "nbd")):
                        block_dev = re.sub(r"p\d+$", "", dev_name)
                    elif dev_name.startswith("loop"):
                        block_dev = dev_name
                    else:
                        block_dev = re.sub(r"\d+$", "", dev_name)
                    if block_dev.startswith("nvme"):
                        info["storage_type"] = "NVMe"
                    else:
                        with open(f"/sys/block/{block_dev}/queue/rotational", encoding="utf-8") as f:
                            info["storage_type"] = "HDD" if f.read().strip() == "1" else "SSD"
            except Exception:
                pass

    except ImportError:
        pass

    # CPU governor and cgroup resource limits (Linux — relevant in containers/HPC)
    if system == "Linux":
        try:
            gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text(encoding="utf-8").strip()
            if gov:
                info["cpu_governor"] = gov
        except Exception:
            pass
        try:
            # cgroups v2: "quota period" or "max max"
            cpu_max = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip()
            if cpu_max and cpu_max != "max max":
                quota, period = cpu_max.split()
                if quota != "max" and period not in ("0", ""):
                    info["cgroup_cpu_limit_cores"] = round(int(quota) / int(period), 2)
        except Exception:
            pass
        try:
            mem_max = Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
            if mem_max and mem_max != "max":
                info["cgroup_memory_limit_gb"] = round(int(mem_max) / (1024 ** 3), 2)
        except Exception:
            pass

    # GPU details via nvidia-smi (works on Linux, macOS, Windows)
    try:
        gpu_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,uuid",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        if gpu_out:
            gpus = []
            for line in gpu_out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                gpu: dict = {}
                if len(parts) > 0 and parts[0]:
                    gpu["name"] = parts[0]
                if len(parts) > 1:
                    try:
                        gpu["memory_mb"] = int(parts[1])
                    except ValueError:
                        pass
                if len(parts) > 2 and parts[2]:
                    gpu["driver_version"] = parts[2]
                if len(parts) > 3 and parts[3]:
                    gpu["uuid"] = parts[3]
                if gpu:
                    gpus.append(gpu)
            if gpus:
                info["gpus"] = gpus
    except Exception:
        pass

    # Env vars that affect GPU selection and threading
    env: dict = {}
    for var in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF"):
        val = os.environ.get(var)
        if val is not None:
            env[var] = val
    if env:
        info["env"] = env

    return info


def collect_runtime_info() -> dict:
    """Return Python version, key package versions, CUDA version, and env hash."""
    import platform as _platform

    rt: dict = {
        "python_version": _platform.python_version(),
        "python_implementation": _platform.python_implementation(),
    }

    for pkg in ("torch", "numpy", "aiohttp"):
        try:
            mod = __import__(pkg)
            rt[pkg] = mod.__version__
        except Exception:
            pass

    # CUDA version from torch is more reliable than nvcc (reflects actual runtime)
    try:
        import torch
        if torch.version.cuda:
            rt["cuda_version"] = torch.version.cuda
    except Exception:
        pass

    # pip freeze hash — short fingerprint to detect silent dependency drift between runs
    try:
        import hashlib
        import subprocess as _sp
        import sys as _sys
        freeze = _sp.check_output(
            [_sys.executable, "-m", "pip", "freeze"],
            stderr=_sp.DEVNULL, text=True, timeout=5,
        )
        rt["env_hash"] = hashlib.sha256(freeze.encode()).hexdigest()[:16]
    except Exception:
        pass

    return rt


def take_pre_run_snapshot() -> dict:
    """Capture available memory and free GPU VRAM immediately before the eval loop."""
    snap: dict = {}
    try:
        import psutil
        mem = psutil.virtual_memory()
        snap["memory_available_gb"] = round(mem.available / (1024 ** 3), 2)
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        free_list = [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
        if free_list:
            snap["gpu_memory_free_mb"] = free_list
    except Exception:
        pass
    return snap


def collect_peak_usage() -> dict:
    """Read OS-tracked peak RSS after the run completes. No sampling thread needed."""
    peak: dict = {}
    try:
        import resource  # Unix only (Linux + macOS)
        import platform as _p
        divisor = 1024 ** 2 if _p.system() == "Darwin" else 1024
        ru_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ru_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak["process_peak_rss_mb"] = round((ru_self + ru_children) / divisor, 1)
    except ImportError:
        # Windows fallback via psutil
        try:
            import psutil
            mem_info = psutil.Process(os.getpid()).memory_info()
            if hasattr(mem_info, "peak_wset"):
                peak["process_peak_rss_mb"] = round(mem_info.peak_wset / (1024 ** 2), 1)
        except Exception:
            pass
    except Exception:
        pass
    return peak
