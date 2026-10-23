"""Build the native CUDA scheduler library with CMake, explicitly or on first use.

Windows builds build/native-cuda/native_cublas.dll with MSVC (Visual Studio 2022 generator); Linux
builds build/native-cuda/native_cublas.so with GCC or Clang (Ninja when installed, else Unix
Makefiles). MF_NATIVE_CMAKE_GENERATOR selects another CMake generator.

``ensure_built`` is what native.py calls before loading the library: it rebuilds when the library is
missing or older than its sources, and a file lock lets concurrent processes (distributed workers,
parallel test runs) wait for one build instead of racing each other. Only the standard library is
imported here, so tools/build_native_cuda.py can run it without CuPy.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]
WINDOWS = os.name == "nt"
OUTPUT = ROOT / "build" / "native-cuda" / ("native_cublas.dll" if WINDOWS else "native_cublas.so")
SOURCES = (PROJECT / "native_cublas.cpp", PROJECT / "CMakeLists.txt")
# SHA-256 of SOURCES that OUTPUT was built from, written after every successful build.
STAMP = OUTPUT.with_name(OUTPUT.name + ".sources")
LOG = OUTPUT.with_name("build.log")
LOCK = OUTPUT.with_name("build.lock")
LOCK_TIMEOUT_S = 1800.0


class BuildError(RuntimeError):
    pass


def source_digest() -> str:
    sha = hashlib.sha256()
    for path in SOURCES:
        sha.update(path.name.encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def stale_reason() -> str | None:
    """Why OUTPUT has to be (re)built, or None when it matches the sources."""
    if not OUTPUT.exists():
        return "missing"
    if STAMP.exists():
        return None if STAMP.read_text(encoding="ascii").strip() == source_digest() else "sources changed"
    # Built before stamps existed: fall back to modification times.
    newest = max(path.stat().st_mtime for path in SOURCES)
    return "sources newer than library" if newest > OUTPUT.stat().st_mtime else None


def find_cmake() -> str:
    # `pip install cmake` into the project environment puts cmake beside the interpreter.
    found = shutil.which("cmake") or shutil.which("cmake", path=str(Path(sys.executable).parent))
    default = Path(r"C:\Program Files\CMake\bin\cmake.exe")
    if not found and WINDOWS and default.exists():
        found = str(default)
    if not found:
        raise BuildError("CMake 3.20+ not found: install it (e.g. apt install cmake, or pip install cmake into .venv)")
    return found


def _commands() -> tuple[list[str], list[str]]:
    cmake = find_cmake()
    default = "Visual Studio 17 2022" if WINDOWS else "Ninja" if shutil.which("ninja") else "Unix Makefiles"
    generator = os.environ.get("MF_NATIVE_CMAKE_GENERATOR", default)
    visual_studio = generator.startswith("Visual Studio")
    # One build directory per generator: CMake refuses to reuse a tree configured by another one.
    build_dir = "build-msvc" if visual_studio else "build-" + re.sub(r"\W+", "-", generator.lower())
    build_tree = OUTPUT.parent / build_dir
    # A DLL is a runtime output and a shared object a library output; both land next to OUTPUT.
    configure = [cmake, "-S", str(PROJECT), "-B", str(build_tree), "-G", generator,
                 f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={OUTPUT.parent}",
                 f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={OUTPUT.parent}"]
    configure += ["-A", "x64"] if visual_studio else ["-DCMAKE_BUILD_TYPE=Release"]
    return configure, [cmake, "--build", str(build_tree), "--config", "Release", "--parallel"]


def build(*, verbose: bool = True) -> Path:
    """Configure and build OUTPUT, then record the source digest. ``verbose`` streams CMake's output;
    otherwise it goes to build/native-cuda/build.log."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    digest = source_digest()
    with LOG.open("w", encoding="utf-8", errors="replace") as log:
        for command in _commands():
            print("$", " ".join(command), file=sys.stdout if verbose else log, flush=True)
            result = subprocess.run(command, cwd=ROOT, stdout=None if verbose else log,
                                    stderr=None if verbose else subprocess.STDOUT, check=False)
            if result.returncode:
                hint = "" if verbose else f"; see {LOG}"
                if WINDOWS and OUTPUT.exists():
                    hint += " (a running process may still hold the old library)"
                raise BuildError(f"{Path(command[0]).name} exited {result.returncode}{hint}")
    multi_config_output = OUTPUT.parent / "Release" / OUTPUT.name
    if multi_config_output.exists():
        shutil.copy2(multi_config_output, OUTPUT)
    if not OUTPUT.exists():
        raise BuildError(f"the build did not produce {OUTPUT}")
    STAMP.write_text(digest + "\n", encoding="ascii")
    return OUTPUT


@contextmanager
def _exclusive(path: Path, timeout_s: float = LOCK_TIMEOUT_S):
    """Hold an OS file lock; it is released when the holder exits, even if it crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    with path.open("a+b") as handle:
        if WINDOWS:
            import msvcrt
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise BuildError(f"timed out waiting for {path}") from None
                    time.sleep(0.5)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if WINDOWS:
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_built() -> bool:
    """Build OUTPUT when it is missing or stale; return whether this call built it."""
    if stale_reason() is None:
        return False
    with _exclusive(LOCK):
        reason = stale_reason()  # another process may have finished the build while we waited
        if reason is None:
            return False
        print(f"[native] building {OUTPUT.name} ({reason}); CMake output: {LOG}", file=sys.stderr, flush=True)
        started = time.perf_counter()
        build(verbose=False)
        print(f"[native] built {OUTPUT} in {time.perf_counter() - started:.1f} s", file=sys.stderr, flush=True)
        return True


def status() -> dict:
    """Library state for diagnostics and the console: path, whether it is current, and why not."""
    try:
        reason = stale_reason()
    except OSError as exc:
        reason = f"sources unreadable: {exc}"
    try:
        cmake = find_cmake()
    except BuildError:
        cmake = None
    return {"path": str(OUTPUT), "built": OUTPUT.exists(), "current": reason is None, "reason": reason,
            "cmake": cmake, "log": str(LOG) if LOG.exists() else None}


if __name__ == "__main__":
    print(build())
