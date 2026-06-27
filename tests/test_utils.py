"""Tests for utility functions - all use lazy imports, no external deps needed."""
import pytest
from whale_video.utils import get_torch, get_device, get_gpu_info, optimize_memory, clear_cache
from whale_video.utils.gpu_utils import get_torch as gpu_get_torch


class TestGetTorch:
    """Test lazy torch import."""

    def test_get_torch_raises_when_not_installed(self):
        """get_torch raises ModuleNotFoundError when torch is not installed."""
        with pytest.raises(ModuleNotFoundError):
            get_torch()

    def test_get_torch_from_gpu_utils_raises_when_not_installed(self):
        """get_torch from gpu_utils also raises when torch is not installed."""
        with pytest.raises(ModuleNotFoundError):
            gpu_get_torch()


class TestGetDevice:
    """Test device detection."""

    def test_get_device_returns_string(self):
        """get_device should always return a string."""
        device = get_device()
        assert isinstance(device, str)
        assert device in ("cuda", "cpu", "mps")

    def test_get_device_no_crash_without_cuda(self):
        """get_device handles missing CUDA gracefully."""
        device = get_device()
        # Should never raise
        assert device is not None


class TestGetGpuInfo:
    """Test GPU info retrieval."""

    def test_get_gpu_info_returns_dict(self):
        """get_gpu_info should always return a dict."""
        info = get_gpu_info()
        assert isinstance(info, dict)

    def test_get_gpu_info_has_available_key(self):
        """get_gpu_info dict should contain 'available' key."""
        info = get_gpu_info()
        assert "available" in info
        assert isinstance(info["available"], bool)


class TestMemoryFunctions:
    """Test memory optimization functions don't crash."""

    def test_optimize_memory_does_not_crash(self):
        """optimize_memory should be safe to call without torch CUDA."""
        # Should not raise any exception
        optimize_memory()

    def test_clear_cache_does_not_crash(self):
        """clear_cache should be safe to call without torch CUDA."""
        # Should not raise any exception
        clear_cache()

    def test_optimize_memory_called_twice(self):
        """Calling optimize_memory multiple times is safe."""
        optimize_memory()
        optimize_memory()
        # No crash expected

    def test_clear_cache_called_twice(self):
        """Calling clear_cache multiple times is safe."""
        clear_cache()
        clear_cache()
        # No crash expected
