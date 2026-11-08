import unittest

import numpy as np

import MyFlows as ms
from MyFlows.core.device import asnumpy, cuda_available, get_device, set_device
from MyFlows.ops.convolution import Conv2D_Op


class RecomputeColumnsTest(unittest.TestCase):
    def check(self, device, backend, groups=1):
        self.addCleanup(set_device, get_device())
        set_device(device)
        rng = np.random.default_rng(100)
        data = [rng.normal(size=s).astype(np.float32) for s in ((2, 4, 13, 11), (6, 4//groups, 3, 3), (6,))]
        states = []
        for recompute in (False, True):
            x, w, b = (ms.Variable(a, trainable=True) for a in data)
            conv = Conv2D_Op(x, w, stride=(1, 2), padding=1, groups=groups, bias=b,
                             backend=backend, recompute_im2col=recompute)
            graph = ms.Graph(ms.ReLU(conv), optimize=True)
            fused = graph.target_node
            self.assertEqual(fused.recompute_im2col, recompute)
            graph.forward()
            cols = fused._cuda_im2col_cols if backend.startswith("cuda_native") else fused.cols
            self.assertEqual(cols is None or isinstance(cols, list) and not cols, recompute)
            # Replacing the leaf's value must not replace the activation saved by the preceding forward pass.
            x.value = np.zeros_like(data[0])
            graph.backward()
            states.append([asnumpy(v).copy() for v in (fused.value, x.grad, w.grad, b.grad)])
        for saved, rebuilt in zip(*states):
            self.assertEqual(saved.tobytes(), rebuilt.tobytes())

    def test_cpu_grouped_columns(self):
        self.check("cpu", "numpy", groups=2)

    @unittest.skipUnless(cuda_available(), "CUDA is required for the three device backends")
    def test_device_backends(self):
        for backend in ("cupy", "cuda_native_register", "cuda_native_cublas"):
            with self.subTest(backend=backend):
                self.check("cuda", backend)
