"""GPU and memory optimization utilities."""


def get_torch():
    import importlib
    return importlib.import_module('torch')


def get_device() -> str:
    """Select best available device."""
    try:
        torch = get_torch()
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_gpu_info() -> dict:
    """Get GPU memory info."""
    try:
        torch = get_torch()
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
        }
    except ImportError:
        return {"available": False}


def optimize_memory():
    """Apply memory optimizations for limited GPU setups."""
    try:
        torch = get_torch()
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def clear_cache():
    """Clear GPU cache."""
    try:
        torch = get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
