"""Compile local CUDA sources with CuPy NVRTC and preserve compilation identity."""

from functools import lru_cache
import hashlib
from pathlib import Path

from ...core.device import _lazy_cupy


OPTIONS = ("--std=c++11",)
SOURCE_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def _module(source, source_hash, options, device_id):
    cp = _lazy_cupy()
    with cp.cuda.Device(device_id):
        return cp.RawModule(code=source, options=options, backend="nvrtc")


def get_kernel(filename, name):
    cp = _lazy_cupy()
    source = (SOURCE_DIR / filename).read_text(encoding="utf-8")
    sha = hashlib.sha256(source.encode()).hexdigest()
    module = _module(source, sha, OPTIONS, cp.cuda.runtime.getDevice())
    return module.get_function(name)


def compilation_metadata():
    cp = _lazy_cupy()
    return {"compiler": "nvrtc", "options": list(OPTIONS),
            "default_block_size": 256, "index_bits": 32,
            "device_id": cp.cuda.runtime.getDevice(),
            "compute_capability": cp.cuda.Device().compute_capability,
            "cupy": cp.__version__, "runtime": cp.cuda.runtime.runtimeGetVersion(),
            "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(SOURCE_DIR.glob("*.cu"))}}
