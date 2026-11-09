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
                self.assertEqual({v["tile"] for v in saved["candidates"] if v["times_ms"]}, set(native.tile_catalog()))
                winner = saved["selected"]
                self.assertEqual(
                    tuned.tobytes(), call(plan=(winner["tile"], winner["requested_slices"])).tobytes()
                )
                autotune._selected.cache_clear()
                with patch.object(autotune, "_calibrate", side_effect=AssertionError("unnecessary repeat")):
                    self.assertEqual(tuned.tobytes(), call().tobytes())

    @unittest.skipUnless(cuda_available(), "extended split reduction requires CUDA")
    def test_narrow_tile_96_splits_and_parallel_reduction_are_deterministic(self):
        self.addCleanup(set_device, get_device())
        set_device("cuda")
        rng=np.random.default_rng(115)
        a,b=[rng.normal(size=(38400,n)).astype(np.float32) for n in (32,16)]
        da,db=xp.asarray(a),xp.asarray(b)
        def run():
            return xp.asnumpy(native.gemm(da,db,trans_a=True,plan=("32x16/4x4/k16",96)))
        first=run()
        self.assertEqual(first.tobytes(),run().tobytes())
        np.testing.assert_allclose(first, a.astype(np.float64).T@b.astype(np.float64), rtol=1e-5, atol=6e-4)

    @unittest.skipUnless(cuda_available(), "compiled tile geometries require CUDA")
    def test_every_measured_ada_plan_names_a_compiled_tile_and_a_launchable_split(self):
        from MyFlows.ops.cuda_native.ada_plans import PLANS

        self.addCleanup(set_device, get_device())
        set_device("cuda")
        catalog = native.tile_catalog()
        for (layout, m, n, k), (tile, requested) in PLANS.items():
            self.assertIn(layout, ("NN", "NT", "TN"))
            self.assertIn(tile, catalog, (layout, m, n, k))
            native._plan_code(tile, requested)
            self.assertLessEqual(native._slices(k, requested, catalog[tile][-1]), requested)
        profile = native._device_profile(xp.cuda.runtime.getDevice())
        if profile:  # the measured table applies only to the 24-SM RTX 4060 Laptop GPU
            with patch.dict(os.environ, {"MYFLOWS_GEMM_AUTOTUNE": "0"}):
                for (layout, m, n, k), (tile, requested) in PLANS.items():
                    expected = (tile, native._slices(k, requested, catalog[tile][-1]))
                    self.assertEqual(native.gemm_plan(m, n, k, layout=layout), expected)

    @unittest.skipUnless(cuda_available(), "compiled tile geometries require CUDA")
    def test_all_compiled_tiles_cover_aligned_and_ragged_operands(self):
        self.addCleanup(set_device, get_device())
        set_device("cuda")
        rng = np.random.default_rng(122)
        for m,n,k in ((17,31,49),(64,32,96)):
            for ta,tb in ((False,False),(True,False),(False,True)):
                a,b = [xp.asarray(rng.normal(size=s).astype(np.float32)) for s in
                       ((k,m) if ta else (m,k),(n,k) if tb else (k,n))]
                def run(tile,splits):
                    return xp.asnumpy(native.gemm(a,b,trans_a=ta,trans_b=tb,plan=(tile,splits)))
                reference=run(native.GEMM_TILES[0],1)
                for tile in native.tile_catalog():
                    self.assertEqual(run(tile,1).tobytes(),reference.tobytes(),(tile,ta,tb))
                    np.testing.assert_allclose(run(tile,3),reference,atol=1.7e-5,rtol=3e-6)
