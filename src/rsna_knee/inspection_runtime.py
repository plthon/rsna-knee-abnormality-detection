import importlib
import importlib.metadata
import os
import platform
import shutil
from pathlib import Path
from typing import Any


def capture_runtime(work_root: Path) -> dict[str, Any]:
    packages = {
        distribution: _distribution_version(distribution)
        for distribution in (
            "numpy",
            "pydicom",
            "pylibjpeg",
            "pylibjpeg-libjpeg",
            "pylibjpeg-openjpeg",
            "python-gdcm",
            "torch",
            "torchvision",
        )
    }
    torch_cuda_available: bool | None = None
    try:
        torch = importlib.import_module("torch")
        torch_cuda_available = bool(torch.cuda.is_available())
    except (ImportError, AttributeError):
        pass

    disk = shutil.disk_usage(work_root)
    return {
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "disk": {"free_bytes": disk.free, "total_bytes": disk.total},
        "kaggle_environment": {
            name: os.environ.get(name)
            for name in (
                "KAGGLE_KERNEL_RUN_TYPE",
                "KAGGLE_KERNEL_INTEGRATIONS",
                "KAGGLE_URL_BASE",
            )
        },
        "memory_bytes": _memory_bytes(),
        "packages": packages,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_cuda_available": torch_cuda_available,
    }


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
