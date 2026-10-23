"""Native library kernels against independent FP64 references (stricter than the CuPy comparison).

A missing native library is built on first use, so only a missing GPU skips these tests.
"""

import unittest

import numpy as np

from MyFlows.core.device import cuda_available, set_device, xp
from MyFlows.core.node import Variable
from MyFlows.ops.convolution import NATIVE_BACKENDS, Conv2D_LeakyReLU_Op, Conv2D_Op, Conv2D_ReLU_Op
from MyFlows.ops.cuda_native import native
from MyFlows.tests.cuda_fixtures import CONV_CASES, conv_fixture
from MyFlows.tests.test_convolution import naive_conv_backward, naive_conv_forward

EPS32 = float(np.finfo(np.float32).eps)

PERFORMANCE_CASES = (
    ("P1", (4, 16, 32, 32), (32, 16, 3, 3), (1, 1), (1, 1)),
    ("P2", (1, 3, 120, 160), (8, 3, 7, 7), (2, 2), (3, 3)),
    ("R1", (2, 33, 9, 11), (40, 33, 3, 3), (1, 1), (1, 1)),  # channel tiles with remainders
)


def _reference(case):
    x, w, b, dy = (a.astype(np.float64) for a in conv_fixture(case, 0))
    stride, padding = case[3], case[4]
    pre = naive_conv_forward(x, w, stride, padding) + b[None, :, None, None]
    return x, w, b, dy, pre


def _assert_near(test, actual, expected, rel, name):
    actual = np.asarray(actual, dtype=np.float64)
    scale = max(float(np.abs(expected).max()), 1e-30)
    error = float(np.abs(actual - expected).max())
    test.assertLessEqual(error, rel * scale, f"{name}: max error {error:.3e} > {rel:.1e} x {scale:.3e}")


@unittest.skipUnless(cuda_available(), "CUDA is required")
class NativeConvolutionKernelTests(unittest.TestCase):
    def setUp(self):
        set_device("cuda")

    def tearDown(self):
        set_device("cpu")

    def _run(self, case, op_class=Conv2D_Op, needs_input_grad=True, backend="cuda_native_register", **op_kwargs):
        x, w, b, dy = (xp.asarray(a) for a in conv_fixture(case, 0))
        x_node = Variable(x, requires_grad=needs_input_grad)
        w_node, b_node = Variable(w, trainable=True), Variable(b, trainable=True)
        op = op_class(x_node, w_node, stride=case[3], padding=case[4], bias=b_node, backend=backend, **op_kwargs)
        op.forward(x, w, b)
        op.grad = dy
        op.backward()
        grads = [None if node.grad is None else xp.asnumpy(node.grad) for node in (x_node, w_node, b_node)]
        return op, xp.asnumpy(op.value), grads

    def test_forward_backward_match_fp64_reference(self):
        for case in CONV_CASES + PERFORMANCE_CASES:
            for backend in NATIVE_BACKENDS:
                with self.subTest(case=case[0], backend=backend):
                    self._check_reference(case, backend)

    def _check_reference(self, case, backend):
        x, w, _, dy, pre = _reference(case)
        op, y, (dx, dw, db) = self._run(case, backend=backend)
        self.assertTrue(op.value.flags.c_contiguous)
        ref_dx, ref_dw = naive_conv_backward(x, w, dy, case[3], case[4])
        _assert_near(self, y, pre, 1e-5, "y")
        _assert_near(self, dx, ref_dx, 1e-5, "dx")
        _assert_near(self, dw, ref_dw, 1e-5, "dw")
        # The bias gradient is accumulated in FP64: at most one float32 rounding away.
        ref_db = dy.sum(axis=(0, 2, 3))
        self.assertTrue(np.all(np.abs(db - ref_db) <= np.spacing(np.abs(ref_db).astype(np.float32))),
                        f"db error {np.abs(db - ref_db).max():.3e} exceeds one ulp")

    def test_fused_relu_and_leaky_relu_match_reference(self):
        for case in (CONV_CASES[1], CONV_CASES[3], PERFORMANCE_CASES[2]):
            x, w, _, dy, pre = _reference(case)
            for (op_class, kwargs, slope), backend in ((variant, backend) for variant in (
                    (Conv2D_ReLU_Op, {}, 0.0), (Conv2D_LeakyReLU_Op, {"alpha": 0.2}, 0.2)) for backend in NATIVE_BACKENDS):
                with self.subTest(case=case[0], op=op_class.__name__, backend=backend):
                    op, y, (dx, dw, db) = self._run(case, op_class, backend=backend, **kwargs)
                    self.assertTrue(op._activation_fused)
                    gate = np.where(pre > 0, 1.0, slope)
                    ref_dx, ref_dw = naive_conv_backward(x, w, dy * gate, case[3], case[4])
                    _assert_near(self, y, np.where(pre > 0, pre, slope * pre), 1e-5, "y")
                    _assert_near(self, dx, ref_dx, 1e-5, "dx")
                    _assert_near(self, dw, ref_dw, 1e-5, "dw")
                    _assert_near(self, db, (dy * gate).sum(axis=(0, 2, 3)), 4 * EPS32, "db")

    def test_input_without_requires_grad_skips_dx(self):
        case = PERFORMANCE_CASES[0]
        for backend in NATIVE_BACKENDS:
            with self.subTest(backend=backend):
                _, _, (_, dw_full, db_full) = self._run(case, backend=backend)
                _, _, (dx, dw, db) = self._run(case, needs_input_grad=False, backend=backend)
                self.assertIsNone(dx)
                np.testing.assert_array_equal(dw, dw_full)
                np.testing.assert_array_equal(db, db_full)

    def test_gradients_are_bitwise_reproducible(self):
        # P1's dW (32 x 144 over K = 4096) takes the register engine's split-K path.
        for case in (PERFORMANCE_CASES[0], PERFORMANCE_CASES[1]):
            for backend in NATIVE_BACKENDS:
                with self.subTest(case=case[0], backend=backend):
                    first = self._run(case, backend=backend)[2]
                    second = self._run(case, backend=backend)[2]
                    for a, b in zip(first, second):
                        np.testing.assert_array_equal(a, b)

    def test_native_errors_name_the_reason(self):
        lib = native._library()
        x = xp.zeros((1, 1, 4, 4), xp.float32)
        status = lib.mf_conv_forward(*(native._u64(x.data.ptr),) * 6,
                                     *(native._i32(v) for v in (1, 1, 4, 4, 1, 3, 3, 9, 9, 1, 1, 0, 0)),
                                     native._i32(0), native.ctypes.c_float(0.0), native._i32(1), native._u64(0),
                                     native.ctypes.c_int64(0), native._u64(0))
        with self.assertRaisesRegex(native.NativeCudaError, r"invalid argument.*native_cublas\.cpp:\d+: .*output size"):
            native._check(status, lib)
        with self.assertRaisesRegex(ValueError, "engine"):
            native.conv2d_forward(x, xp.zeros((1, 1, 3, 3), xp.float32), engine="tf32")
        with self.assertRaisesRegex(ValueError, "int32"):
            native._guard_index(columns=2**31)
        with self.assertRaisesRegex(ValueError, "alpha > 0"):
            native.conv2d_forward(x, xp.zeros((1, 1, 3, 3), xp.float32), activation="leaky_relu", alpha=0.0)

    def test_debug_sync_and_build_info(self):
        info = native.info()
        self.assertEqual(info["device_arch"], int(xp.cuda.Device().compute_capability))
        self.assertIn(info["binary"], ("cubin", "ptx"))
        native.set_debug_sync(True)
        try:
            self._run(CONV_CASES[1])
        finally:
            native.set_debug_sync(False)


@unittest.skipUnless(cuda_available(), "CUDA is required")
class NativeMatmulTests(unittest.TestCase):
    """Lecture 19 matmul kernels: basic, tiled, shared-memory and register-tiled, three accumulation modes."""

    def setUp(self):
        set_device("cuda")
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        set_device("cpu")

    def _operands(self, m, n, k):
        a = self.rng.standard_normal((m, k)).astype(np.float32)
        b = self.rng.standard_normal((k, n)).astype(np.float32)
        return a, b, a.astype(np.float64) @ b.astype(np.float64), np.abs(a).astype(np.float64) @ np.abs(b)

    def _all_kernels(self):
        for variant in ("basic", "tiled", "shared", "register"):
            for tile in ((32,) if variant in ("basic", "register") else (16, 32)):
                for accumulate in native.MATMUL_ACCUMULATION:
                    yield variant, tile, accumulate

    def test_lecture_8x8_case_is_within_1e_5_of_numpy(self):
        a, b, _, _ = self._operands(8, 8, 8)
        expected = a @ b
        for variant, tile, accumulate in list(self._all_kernels()) + [("cublas", 32, "fp32")]:
            got = xp.asnumpy(native.matmul(xp.asarray(a), xp.asarray(b), variant=variant, tile=tile, accumulate=accumulate))
            self.assertLess(float(np.abs(got - expected).max()), 1e-5, (variant, tile, accumulate))

    def test_error_bounds_on_ragged_shapes(self):
        for m, n, k in ((1, 1, 1), (67, 39, 45), (33, 65, 300), (5, 129, 1000)):
            a, b, exact, magnitude = self._operands(m, n, k)
            gamma = k * EPS32 / 2 / (1 - k * EPS32 / 2)  # classical worst-case dot-product bound
            for variant, tile, accumulate in self._all_kernels():
                with self.subTest(shape=(m, n, k), kernel=(variant, tile, accumulate)):
                    got = xp.asnumpy(native.matmul(xp.asarray(a), xp.asarray(b), variant=variant,
                                                   tile=tile, accumulate=accumulate)).astype(np.float64)
                    error = np.abs(got - exact)
                    self.assertTrue(np.all(error <= gamma * magnitude + 1e-30))
                    if accumulate == "dot2":
                        # Twice-the-working-precision accumulation: within one ulp of the exact value.
                        self.assertTrue(np.all(error <= np.spacing(np.abs(exact).astype(np.float32)) + gamma ** 2 * magnitude))

    def test_compensated_modes_beat_plain_fp32_at_large_k(self):
        a, b, exact, _ = self._operands(64, 64, 4096)
        errors = {}
        for accumulate in native.MATMUL_ACCUMULATION:
            got = xp.asnumpy(native.matmul(xp.asarray(a), xp.asarray(b), variant="shared", tile=32, accumulate=accumulate))
            errors[accumulate] = float(np.sqrt(np.mean((got - exact) ** 2)))
        self.assertLess(errors["kahan"], errors["fp32"] / 2)
        self.assertLess(errors["dot2"], errors["kahan"])

    def test_register_tiling_matches_basic_bitwise(self):
        # Same summation order (kahan groups of 32 terms) as the basic kernel. 256 x 132 x 68 takes the
        # float4 path; 67 x 39 x 45 and the operands placed 4 bytes past an aligned address take the
        # scalar one (a misaligned float4 load would fault).
        for m, n, k, offset in ((256, 132, 68, 0), (300, 260, 1000, 0), (67, 39, 45, 0), (130, 132, 68, 1)):
            a, b, _, _ = self._operands(m, n, k)
            da = xp.empty(m * k + offset, xp.float32)[offset:].reshape(m, k)
            db = xp.empty(k * n + offset, xp.float32)[offset:].reshape(k, n)
            da[...], db[...] = xp.asarray(a), xp.asarray(b)
            for accumulate in native.MATMUL_ACCUMULATION:
                with self.subTest(shape=(m, n, k), offset=offset, accumulate=accumulate):
                    got = native.matmul(da, db, variant="register", accumulate=accumulate)
                    expected = native.matmul(da, db, variant="basic", accumulate=accumulate)
                    self.assertTrue(bool(xp.array_equal(got, expected)))

    def test_bias_epilogue_and_cublas(self):
        a, b, exact, _ = self._operands(19, 23, 31)
        bias = self.rng.standard_normal(23).astype(np.float32)
        for variant in ("shared", "register", "cublas"):
            got = xp.asnumpy(native.matmul(xp.asarray(a), xp.asarray(b), xp.asarray(bias), variant=variant))
            np.testing.assert_allclose(got, exact + bias, rtol=0, atol=1e-5)

    def test_invalid_arguments_raise(self):
        a = xp.zeros((2, 3), xp.float32)
        with self.assertRaises(ValueError):
            native.matmul(a, xp.zeros((4, 2), xp.float32))
        with self.assertRaises(ValueError):
            native.matmul(a, xp.zeros((3, 2), xp.float32), tile=8)
        with self.assertRaises(TypeError):
            native.matmul(a.astype(xp.float64), xp.zeros((3, 2), xp.float32))


@unittest.skipUnless(cuda_available(), "CUDA is required")
class NativeGemmEngineTests(unittest.TestCase):
    """native.gemm: register-tiled engine (five tile shapes, NN/NT/TN, split-K) and cuBLAS."""

    # Small outputs over a long K take the split-K path (36 x 20 x 3000, 16 x 144 x 19200); 76800 x 16
    # and 64 x 576 use the tall and wide tiles; 67 x 39 x 45 is ragged everywhere.
    SHAPES = ((67, 39, 45), (256, 132, 68), (300, 260, 1000), (36, 20, 3000), (16, 144, 19200),
              (4096, 16, 147), (64, 576, 2048), (1, 1, 1))
    LAYOUTS = ((False, False), (False, True), (True, False))

    def setUp(self):
        set_device("cuda")
        self.rng = np.random.default_rng(2)

    def tearDown(self):
        set_device("cpu")

    def _operands(self, m, n, k, trans_a, trans_b):
        a = self.rng.standard_normal((k, m) if trans_a else (m, k)).astype(np.float32)
        b = self.rng.standard_normal((n, k) if trans_b else (k, n)).astype(np.float32)
        op_a = (a.T if trans_a else a).astype(np.float64)
        op_b = (b.T if trans_b else b).astype(np.float64)
        return a, b, op_a @ op_b, np.abs(op_a) @ np.abs(op_b)

    def test_error_bounds_and_determinism(self):
        for (m, n, k), (trans_a, trans_b) in ((s, t) for s in self.SHAPES for t in self.LAYOUTS):
            a, b, exact, magnitude = self._operands(m, n, k, trans_a, trans_b)
            bias = self.rng.standard_normal(n).astype(np.float32)
            # Any summation order stays within gamma_K; the split-K reduction and the bias add are one
            # rounding each.
            gamma = k * EPS32 / 2 / (1 - k * EPS32 / 2)
            for engine in native.GEMM_ENGINES:
                with self.subTest(shape=(m, n, k), trans=(trans_a, trans_b), engine=engine):
                    run = lambda: native.gemm(xp.asarray(a), xp.asarray(b), xp.asarray(bias), trans_a=trans_a,
                                              trans_b=trans_b, engine=engine)
                    got = run()
                    self.assertTrue(bool(xp.array_equal(got, run())))
                    error = np.abs(xp.asnumpy(got).astype(np.float64) - (exact + bias))
                    bound = gamma * magnitude + EPS32 * (np.abs(exact) + np.abs(exact + bias)) + 1e-30
                    self.assertTrue(np.all(error <= bound), f"max error {error.max():.3e}")

    def test_unsplit_register_engine_matches_basic_bitwise(self):
        # K < 128 is never split: one pass over K in order on each tile shape (s, h, t, w, x) and layout,
        # with aligned (float4) and misaligned (scalar) operands.
        lib = native._library()
        shapes = ((300, 260, 100), (300, 64, 100), (4096, 16, 100), (64, 576, 100), (32, 300, 100), (67, 39, 45))
        for (m, n, k), (trans_a, trans_b) in ((s, t) for s in shapes for t in self.LAYOUTS):
            self.assertEqual(lib.mf_gemm_workspace(m, n, k), 0)
            a, b, _, _ = self._operands(m, n, k, trans_a, trans_b)
            for offset in (0, 1):
                with self.subTest(shape=(m, n, k), trans=(trans_a, trans_b), offset=offset):
                    da = xp.empty(a.size + offset, xp.float32)[offset:].reshape(a.shape)
                    db = xp.empty(b.size + offset, xp.float32)[offset:].reshape(b.shape)
                    da[...], db[...] = xp.asarray(a), xp.asarray(b)
                    got = native.gemm(da, db, trans_a=trans_a, trans_b=trans_b)
                    expected = native.matmul(xp.ascontiguousarray(da.T if trans_a else da),
                                             xp.ascontiguousarray(db.T if trans_b else db),
                                             variant="basic", accumulate="fp32")
                    self.assertTrue(bool(xp.array_equal(got, expected)))

    def test_split_k_workspace(self):
        lib = native._library()
        self.assertGreater(lib.mf_gemm_workspace(36, 20, 3000), 0)
        self.assertEqual(lib.mf_gemm_workspace(4096, 4096, 1024), 0)
        # 36 x 20 is one tile; K = 3000 is split, the reduction is in FP64.
        a, b, exact, _ = self._operands(36, 20, 3000, False, False)
        split = xp.asnumpy(native.gemm(xp.asarray(a), xp.asarray(b))).astype(np.float64)
        one_pass = xp.asnumpy(native.matmul(xp.asarray(a), xp.asarray(b), variant="basic", accumulate="fp32"))
        self.assertLess(np.abs(split - exact).max(), np.abs(one_pass - exact).max())

    def test_invalid_arguments_raise(self):
        a = xp.zeros((2, 3), xp.float32)
        with self.assertRaises(ValueError):
            native.gemm(a, xp.zeros((2, 3), xp.float32))
        with self.assertRaises(ValueError):
            native.gemm(a, a, trans_a=True, trans_b=True)
        with self.assertRaises(ValueError):
            native.gemm(a, xp.zeros((3, 2), xp.float32), engine="tf32")
        with self.assertRaises(TypeError):
            native.gemm(a.astype(xp.float64), xp.zeros((3, 2), xp.float32))


@unittest.skipUnless(cuda_available(), "CUDA is required")
class NativeLinearTests(unittest.TestCase):
    """Linear/Dense on the native backends: forward, dX, dW and db against FP64."""

    def setUp(self):
        set_device("cuda")
        rng = np.random.default_rng(1)
        self.x, self.w = rng.standard_normal((37, 300), dtype=np.float32), rng.standard_normal((300, 23), dtype=np.float32)
        self.b, self.g = rng.standard_normal((1, 23), dtype=np.float32), rng.standard_normal((37, 23), dtype=np.float32)

    def tearDown(self):
        set_device("cpu")

    def _run(self, backend, x_requires_grad=True):
        from MyFlows.ops.basic import Linear

        nodes = [Variable(xp.asarray(self.x), requires_grad=x_requires_grad),
                 Variable(xp.asarray(self.w), trainable=True), Variable(xp.asarray(self.b), trainable=True)]
        op = Linear(*nodes, backend=backend)
        op.forward(*(n.value for n in nodes))
        for n in nodes:
            n.clear_grad()
        op.grad = xp.asarray(self.g)
        op.backward()
        return [xp.asnumpy(op.value)] + [xp.asnumpy(n.grad) for n in nodes]

    def test_backends_match_fp64(self):
        x, w, b, g = (a.astype(np.float64) for a in (self.x, self.w, self.b, self.g))
        expected = (x @ w + b, g @ w.T, x.T @ g, g.sum(axis=0, keepdims=True))
        magnitude = (np.abs(x) @ np.abs(w) + np.abs(b), np.abs(g) @ np.abs(w.T), np.abs(x.T) @ np.abs(g))
        gamma = 300 * EPS32 / 2 / (1 - 300 * EPS32 / 2)
        for backend in ("auto", "cuda_native_register", "cuda_native_cublas", "cuda_native_kahan", "cuda_native_dot2"):
            with self.subTest(backend=backend):
                got = self._run(backend)
                for actual, exact, bound in zip(got[:3], expected[:3], magnitude):
                    error = np.abs(actual - exact)
                    self.assertTrue(np.all(error <= gamma * bound + 1e-30))
                    if backend == "cuda_native_dot2":
                        # The bias add after the dot product is one more rounding.
                        self.assertTrue(np.all(error <= 2 * np.spacing(np.abs(exact).astype(np.float32)) + gamma ** 2 * bound))
                # db is accumulated in FP64 on every backend: one rounding away from the exact sum.
                self.assertTrue(np.all(np.abs(got[3] - expected[3]) <= np.spacing(np.abs(expected[3]).astype(np.float32))))
        errors = {backend: float(np.sqrt(np.mean((self._run(backend)[2] - expected[2]) ** 2)))
                  for backend in ("cuda_native_register", "cuda_native_cublas", "cuda_native_kahan", "cuda_native_dot2")}
        self.assertLess(errors["cuda_native_dot2"], errors["cuda_native_kahan"])

    def test_input_without_requires_grad_skips_dx(self):
        _, dx, dw, _ = self._run("cuda_native_kahan", x_requires_grad=False)
        self.assertFalse(np.any(dx))
        np.testing.assert_array_equal(dw, self._run("cuda_native_kahan")[2])

    def test_dense_layer_and_backend_validation(self):
        import MyFlows as ms

        for backend in ("cuda_native_register", "cuda_native_kahan"):
            layer = ms.Dense(300, 23, backend=backend)
            node = layer(Variable(xp.asarray(self.x), requires_grad=False))
            node.forward(*(p.value for p in node.parents))
            self.assertEqual(node.backend, backend)
        with self.assertRaises(ValueError):
            ms.Dense(3, 2, backend="tf32")
        set_device("cpu")
        op = ms.Dense(3, 2, backend="cuda_native_cublas")(Variable(np.zeros((1, 3), np.float32)))
        with self.assertRaises(ValueError):
            op.forward(*(p.value for p in op.parents))


if __name__ == "__main__":
    unittest.main()
