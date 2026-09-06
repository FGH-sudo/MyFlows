# -*- coding: utf-8 -*-
import sys
import unittest
import random
import time
from pathlib import Path


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
  sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows.data.pipeline import MultiprocessDataLoader


def _load_pipeline_sample(value):
  return value, value


def _load_uneven_sample(value):
  if value % 3 == 0:
    time.sleep(0.01)
  return value, value


def _load_broken_sample(value):
  raise ValueError("injected load failure")


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
    self.assertEqual(flat, list(range(20)))
    self.assertEqual([len(batch) for batch, _ in batches], [4] * 5)

  def test_worker_scheduling_preserves_seeded_batches_and_single_tail(self):
    expected = list(range(21))
    random.Random(17).shuffle(expected)
    loader = MultiprocessDataLoader(list(range(21)), batch_size=4, num_workers=3,
                                   shuffle=True, seed=17, load_fn=_load_uneven_sample)
    batches = list(loader)
    self.assertEqual([len(batch) for batch, _ in batches], [4, 4, 4, 4, 4, 1])
    self.assertEqual([item for batch, _ in batches for item in batch], expected)
    self.assertEqual([item for _, meta in batches for item in meta], expected)

  def test_worker_load_error_is_reported(self):
    loader = MultiprocessDataLoader([1, 2], batch_size=1, num_workers=2, load_fn=_load_broken_sample)
    with self.assertRaisesRegex(RuntimeError, "injected load failure"):
      list(loader)


if __name__ == "__main__":
  unittest.main()
