# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

from MyFlows.utils.tensorboard_logger import TensorBoardLogger, has_tensorboard_events


class TensorBoardLoggerTest(unittest.TestCase):
    def test_write_scalar_creates_events(self):
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError:
            self.skipTest("tensorboard/torch not installed")

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run"
            with TensorBoardLogger(log_dir) as tb:
                self.assertTrue(tb.active)
                tb.log_scalar("train/loss", 0.5, 1)
                tb.log_scalar("train/loss", 0.3, 2)
                tb.flush()
            self.assertTrue(has_tensorboard_events(log_dir))

    def test_disabled_logger(self):
        with tempfile.TemporaryDirectory() as tmp:
            tb = TensorBoardLogger(Path(tmp) / "off", enabled=False)
            self.assertFalse(tb.active)
            tb.log_scalar("x", 1.0, 0)
            tb.close()


if __name__ == "__main__":
    unittest.main()
