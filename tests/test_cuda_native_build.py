"""On-demand build of the native library: staleness, the build lock and the disabled path."""

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from MyFlows.ops.cuda_native import build, native


class NativeBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.sources = (root / "native_cublas.cpp", root / "CMakeLists.txt")
        for path in self.sources:
            path.write_text(path.name, encoding="utf-8")
        output = root / "out" / "native_cublas.dll"
        self.patches = [
            mock.patch.object(build, "SOURCES", self.sources),
            mock.patch.object(build, "OUTPUT", output),
            mock.patch.object(build, "STAMP", output.with_name(output.name + ".sources")),
            mock.patch.object(build, "LOCK", output.with_name("build.lock")),
            mock.patch.object(build, "LOG", output.with_name("build.log")),
        ]
        for patch in self.patches:
            patch.start()
        self.builds = []

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.tmp.cleanup()

    def _fake_build(self, verbose=False):
        self.builds.append(threading.get_ident())
        time.sleep(0.2)
        build.OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        build.OUTPUT.write_bytes(b"library")
        build.STAMP.write_text(build.source_digest(), encoding="ascii")
        return build.OUTPUT

    def test_stale_reason_follows_the_source_stamp(self):
        self.assertEqual(build.stale_reason(), "missing")
        with mock.patch.object(build, "build", self._fake_build):
            self.assertTrue(build.ensure_built())
            self.assertIsNone(build.stale_reason())
            self.assertFalse(build.ensure_built())
            self.sources[0].write_text("changed kernel", encoding="utf-8")
            self.assertEqual(build.stale_reason(), "sources changed")
            self.assertTrue(build.ensure_built())
        self.assertEqual(len(self.builds), 2)

    def test_library_without_stamp_compares_modification_times(self):
        build.OUTPUT.parent.mkdir(parents=True)
        build.OUTPUT.write_bytes(b"old library")
        now = time.time()
        os.utime(build.OUTPUT, (now, now))
        for path in self.sources:
            os.utime(path, (now - 60, now - 60))
        self.assertIsNone(build.stale_reason())
        os.utime(self.sources[0], (now + 60, now + 60))
        self.assertEqual(build.stale_reason(), "sources newer than library")

    def test_concurrent_callers_build_once(self):
        with mock.patch.object(build, "build", self._fake_build):
            threads = [threading.Thread(target=build.ensure_built) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        self.assertEqual(len(self.builds), 1)
        self.assertIsNone(build.stale_reason())

    def test_failed_build_names_the_toolchain(self):
        with mock.patch.object(build, "build", side_effect=build.BuildError("cmake exited 1")), \
                mock.patch.object(native, "LIBRARY_PATH", build.OUTPUT), \
                self.assertRaisesRegex(native.NativeCudaError, "CMake and a C\\+\\+ compiler"):
            native.ensure_library()

    def test_disabled_autobuild_reports_a_missing_library(self):
        with mock.patch.dict(os.environ, {"MYFLOWS_NATIVE_AUTOBUILD": "0"}), \
                mock.patch.object(native, "LIBRARY_PATH", build.OUTPUT), \
                mock.patch.object(build, "build", self._fake_build), \
                self.assertRaisesRegex(native.NativeCudaError, "MYFLOWS_NATIVE_AUTOBUILD=0"):
            native.ensure_library()
        self.assertEqual(self.builds, [])


if __name__ == "__main__":
    unittest.main()
