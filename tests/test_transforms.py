# -*- coding: utf-8 -*-
import unittest

import numpy as np

from MyFlows.utils.transforms import (
    ColorJitter,
    ComposeTransform,
    CutMix,
    MixUp,
    RandomCrop,
    RandomRotation,
    apply_batch_pairwise_mix,
    chw_to_hwc,
    hwc_to_chw,
)


class TransformsTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.img = np.random.rand(32, 48, 3).astype(np.float32)
        self.label = np.array([0.1, 0.5], dtype=np.float64)

    def test_compose_preserves_shape(self):
        t = ComposeTransform([
            RandomCrop(seed=0),
            RandomRotation(seed=0),
            ColorJitter(seed=0),
        ])
        out, lab = t(self.img.copy(), self.label.copy())
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(lab.shape, self.label.shape)

    def test_mixup_label_blend(self):
        partner = np.ones_like(self.img) * 0.8
        partner_lab = np.array([0.9, 0.1])
        m = MixUp(alpha=0.4, seed=0)
        m.set_partner(partner, partner_lab)
        _, lab = m(self.img.copy(), self.label.copy())
        self.assertEqual(lab.shape, (2,))
        self.assertTrue(np.all(lab >= 0) and np.all(lab <= 1))

    def test_cutmix_area_blend(self):
        partner = np.zeros_like(self.img)
        partner_lab = np.array([1.0, 0.0])
        c = CutMix(alpha=1.0, seed=1)
        c.set_partner(partner, partner_lab)
        out, lab = c(self.img.copy(), self.label.copy())
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(lab.shape, (2,))

    def test_batch_mixup_reproducible_with_seed(self):
        imgs = [self.img.copy(), partner := np.random.rand(32, 48, 3).astype(np.float32)]
        labs = [self.label.copy(), np.array([0.2, 0.8])]
        a1, l1 = apply_batch_pairwise_mix(imgs, labs, use_mixup=True, use_cutmix=False, seed=7)
        a2, l2 = apply_batch_pairwise_mix(imgs, labs, use_mixup=True, use_cutmix=False, seed=7)
        np.testing.assert_allclose(l1[0], l2[0])
        np.testing.assert_allclose(a1[0], a2[0])

    def test_chw_hwc_roundtrip(self):
        chw = np.transpose(self.img, (2, 0, 1))
        hwc = chw_to_hwc(chw)
        back = hwc_to_chw(hwc)
        np.testing.assert_allclose(chw, back)


if __name__ == "__main__":
    unittest.main()
