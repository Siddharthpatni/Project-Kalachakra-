"""Domain 43 — Scientific Reproducibility: Deterministic Seeding."""

import os
import random
from typing import Any

import numpy as np

from kalachakra.core.logging import get_logger

log = get_logger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seeds across all frameworks.

    Args:
        seed: Random seed value. Must be non-negative.

    Sets seeds for:
        - Python's random module
        - NumPy
        - PyTorch (if available)
        - CUDA (if available)
        - Environment variables for hash seed
    """
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (optional dependency)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
    except ImportError:
        pass

    log.info(f"Global seed set to {seed}")


def get_reproducibility_info() -> dict[str, Any]:
    """Capture full environment snapshot for reproducibility.

    Returns:
        Dictionary with Python version, package versions, OS info,
        seed state, and hardware details.
    """
    import platform
    import sys

    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "numpy_version": np.__version__,
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "not installed"

    try:
        import sklearn

        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass

    return info
