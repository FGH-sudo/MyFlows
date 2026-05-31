# -*- coding: utf-8 -*-
import unittest

from MyFlows.data.pipeline import MultiprocessDataLoader


class TestPipeline(unittest.TestCase):
  def test_iter_requires_load_fn(self):
    loader = MultiprocessDataLoader([1, 2, 3], batch_size=2, load_fn=None)
    with self.assertRaises(ValueError):
      next(iter(loader))


if __name__ == "__main__":
  unittest.main()
