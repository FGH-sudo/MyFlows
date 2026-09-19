"""Task-facing helpers and the single-process numeric reference."""

from .schedule import run_local_training
from .session import TrainSession
from .tasks.synthetic import make_batches


def single_process(seed=0, global_batch=32, steps=20, optimizer="mbgd", learning_rate=0.01, device="cpu"):
    result = run_local_training({
        "task": "synthetic",
        "seed": seed,
        "global_batch": global_batch,
        "steps": steps,
        "epochs": 1,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "device": device,
        "train_workers": 1,
        "collect_history": True,
    })
    return {"history": result["history"], "losses": result["losses"]}


class SmallMLP:
    """Compatibility wrapper around the synthetic TrainSession."""

    def __init__(self, seed=0):
        self.session = TrainSession({
            "task": "synthetic", "seed": seed, "optimizer": "mbgd",
            "learning_rate": 0.01, "device": "cpu", "train_workers": 1,
            "global_batch": 32, "steps": 1,
        })
        self.params = self.session.params
        self.optimizer = self.session.optimizer

    def snapshot(self):
        return self.session.named_parameters_cpu()

    def load(self, params):
        self.session.load_parameters(params)

    def gradients(self, x, y):
        computed = self.session.forward_backward(x, y)
        return computed["loss"], computed["gradients"]

    def update(self, gradients):
        self.session.apply_global_gradients(gradients, f"local-{self.session.local_version}")
