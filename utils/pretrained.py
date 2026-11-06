"""Compatibility exports for the transfer-learning components."""

from ..transfer.registry import ARCHS, TORCHVISION_WEIGHTS, build_model, PretrainedModelRegistry
from ..transfer.loader import (
    IGNORED_SUFFIX, NUM_BATCHES_TRACKED_POLICY, CheckpointLoader, PretrainedLoadError,
    file_sha256, read_manifest, load_torchvision_weights, summarize_load_report,
    _assign, _target_name, _target_shape,
)
from ..transfer.freeze import (
    ParameterFreezer, freeze_stages, unfreeze_all, freeze_summary,
    frozen_tensors, frozen_fingerprint, _stage_layers,
)
from ..transfer.manager import TransferLearningManager
from ..transfer.heads import HeadAdapterBuilder
from ..transfer.adapters import FeatureAdapter
