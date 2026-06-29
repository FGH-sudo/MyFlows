# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
  sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows.data.pipeline import MultiprocessDataLoader


def _load_pipeline_sample(value):
  return value, value


class TestPipeline(unittest.TestCase):
  def test_iter_requires_load_fn(self):
    loader = MultiprocessDataLoader([1, 2, 3], batch_size=2, load_fn=None)
    with self.assertRaises(ValueError):
      next(iter(loader))

  def test_multiprocess_loader_handles_more_samples_than_queue_window(self):
    loader = MultiprocessDataLoader(
        list(range(20)),
        batch_size=4,
        num_workers=2,
        shuffle=False,
        load_fn=_load_pipeline_sample,
    )
    batches = list(loader)
    self.assertEqual(len(batches), 5)
    flat = [item for batch, _ in batches for item in batch]
    self.assertEqual(sorted(flat), list(range(20)))


if __name__ == "__main__":
  unittest.main()
