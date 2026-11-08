import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from MyFlows.core.device import cuda_available, get_device, set_device, xp
from MyFlows.ops.cuda_native import autotune, native


class RegisterPlanTest(unittest.TestCase):
    def test_explicit_plan_rejects_unknown_tiles_or_slices(self):
        for tile, slices in (("invalid", 1), ("128x128", 0), ("128x64", 33)):
            with self.assertRaises(ValueError):
                native._plan_code(tile, slices)
        self.assertEqual(native._slices(33, 2), 2)

    @unittest.skipUnless(cuda_available(), "tile measurements require CUDA")
    def test_all_tiles_and_layouts_rounding_and_persistent_winner(self):
        self.addCleanup(set_device, get_device())
        set_device("cuda")
        rng = np.random.default_rng(41)
        m, n, k = 19, 37, 67
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, {"MYFLOWS_GEMM_AUTOTUNE": "1", "MYFLOWS_AUTOTUNE_CACHE": folder}),
        ):
            for layout in ("NN", "TN", "NT"):
                ta, tb = layout == "TN", layout == "NT"
                a = xp.asarray(rng.normal(size=(k, m) if ta else (m, k)).astype(np.float32))
                b = xp.asarray(rng.normal(size=(n, k) if tb else (k, n)).astype(np.float32))
                call = lambda **kw: xp.asnumpy(native.gemm(a, b, trans_a=ta, trans_b=tb, **kw))
                base = call(plan=(native.GEMM_TILES[0], 1))
                for tile in native.GEMM_TILES:
                    self.assertEqual(base.tobytes(), call(plan=(tile, 1)).tobytes())
                    np.testing.assert_allclose(call(plan=(tile, 3)), base, atol=8e-6, rtol=4e-6)
                tuned = call()
                saved = json.loads(autotune.cache_file(layout, m, n, k).read_text(encoding="utf-8"))
                self.assertEqual(len({v["tile"] for v in saved["candidates"] if v["times_ms"]}), 5)
                winner = saved["selected"]
                self.assertEqual(
                    tuned.tobytes(), call(plan=(winner["tile"], winner["requested_slices"])).tobytes()
                )
                autotune._selected.cache_clear()
                with patch.object(autotune, "_calibrate", side_effect=AssertionError("unnecessary repeat")):
                    self.assertEqual(tuned.tobytes(), call().tobytes())
