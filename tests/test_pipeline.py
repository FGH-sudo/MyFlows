# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
  sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows.data.pipeline import MultiprocessDataLoader


class TestPipeline(unittest.TestCase):
  def test_iter_requires_load_fn(self):
    loader = MultiprocessDataLoader([1, 2, 3], batch_size=2, load_fn=None)
    with self.assertRaises(ValueError):
      next(iter(loader))


if __name__ == "__main__":
  unittest.main()
