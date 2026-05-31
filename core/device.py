# -*- coding: utf-8 -*-
"""计算设备管理：CPU (NumPy) 与 CUDA (CuPy)。"""

from __future__ import annotations

import numpy as np

_DEVICE = "cpu"
_CUPY = None
_CUDA_PATH_CONFIGURED = False


def configure_cuda_dll_path() -> None:
    """将 CUDA/cuDNN 等 DLL 目录加入进程（供 CuPy、ONNX Runtime GPU 加载）。"""
    _configure_cuda_lib_path()


def _configure_cuda_lib_path() -> None:
    """Windows 上优先复用 PyTorch 自带的 CUDA 动态库目录。"""
    global _CUDA_PATH_CONFIGURED
    if _CUDA_PATH_CONFIGURED:
        return
    _CUDA_PATH_CONFIGURED = True

    import os
    import sys

    candidates = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(os.path.join(cuda_path, "bin"))
    try:
        import torch

        candidates.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except ImportError:
        pass

    if sys.platform == "win32":
        for directory in candidates:
            if directory and os.path.isdir(directory):
                try:
                    os.add_dll_directory(directory)
                except (AttributeError, OSError):
                    os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
    else:
        for directory in candidates:
            if directory and os.path.isdir(directory):
                os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")


class _ArrayModuleProxy:
    """动态转发到当前设备对应的数组库（np 或 cupy）。"""

    def __getattr__(self, name):
        return getattr(_active_module(), name)


xp = _ArrayModuleProxy()


def _lazy_cupy():
    global _CUPY
    if _CUPY is None:
        _configure_cuda_lib_path()
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError(
                "GPU 加速需要 CuPy。请安装与 CUDA 版本匹配的包，例如：\n"
                "  pip install cupy-cuda12x\n"
                "详见 requirements-gpu.txt"
            ) from exc
        _CUPY = cp
    return _CUPY


def cuda_available() -> bool:
    """检测 CuPy 是否能在本机 GPU 上完成一次简单运算。"""
    try:
        cp = _lazy_cupy()
        if not cp.cuda.is_available():
            return False
        _ = cp.asarray([1.0], dtype=cp.float32) + 1
        return True
    except Exception:
        return False


def _active_module():
    if _DEVICE == "cuda":
        return _lazy_cupy()
    return np


def get_device() -> str:
    return _DEVICE


def set_device(device: str) -> str:
    """切换全局计算设备，返回实际生效的设备名。"""
    global _DEVICE
    name = str(device).strip().lower()
    if name in ("cpu",):
        _DEVICE = "cpu"
        return _DEVICE
    if name in ("cuda", "gpu", "cuda:0"):
        cp = _lazy_cupy()
        if not cp.cuda.is_available():
            raise RuntimeError("已请求 CUDA，但当前环境检测不到可用 GPU。")
        _DEVICE = "cuda"
        return _DEVICE
    raise ValueError(f"不支持的设备: {device!r}，请使用 'cpu' 或 'cuda'")


def use_cuda(device: str = "cuda") -> str:
    """在 CUDA 可用时切换到 GPU，否则保持 CPU。"""
    if cuda_available():
        return set_device(device)
    return set_device("cpu")


def is_gpu_array(arr) -> bool:
    if _DEVICE != "cuda":
        return False
    try:
        cp = _lazy_cupy()
    except ImportError:
        return False
    return isinstance(arr, cp.ndarray)


def asnumpy(arr, *, copy: bool = False) -> np.ndarray:
    """将数组安全地拷贝到 CPU NumPy。"""
    if arr is None:
        return None
    if is_gpu_array(arr):
        out = _lazy_cupy().asnumpy(arr)
        return out.copy() if copy else out
    data = np.asarray(arr)
    return data.copy() if copy else data


def to_device(data):
    """将数据放到当前全局设备上（返回 Tensor）。"""
    from .tensor import Tensor

    return Tensor.ensure(data)


def module_name() -> str:
    return _active_module().__name__
