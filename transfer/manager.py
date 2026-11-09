"""Construct and configure one transfer task while preserving the original linear head by default."""

from __future__ import annotations

import math
import hashlib
import numpy as np

from ..core.device import asnumpy
from ..utils.checkpoint import named_state

from .adapters import FeatureAdapter
from .freeze import ParameterFreezer, _stage_layers, freeze_summary, parse_schedule
from .heads import HeadAdapterBuilder
from .loader import CheckpointLoader, summarize_load_report
from .registry import PretrainedModelRegistry


def parse_lr_multipliers(value):
    if not value:
        return {}
    try:
        items = [(name, float(rate)) for name, rate in (entry.split(":") for entry in value.split(","))]
    except (ValueError, TypeError) as exc:
        raise ValueError("stage LR multipliers must have the form stem:0.01,layer1:0.1") from exc
    if len(dict(items)) != len(items):
        raise ValueError("stage LR names must be unique")
    return dict(items)


class TransferLearningManager:
    def __init__(self, model, *, freeze_stages=0, freeze_mode="both", unfreeze_schedule=None, bn_unfreeze="follow",
                 backbone_lr_mult=1., stage_lr_mults=None, load_report=None):
        self.model = model
        self.freezer = ParameterFreezer(model)
        self.initial_level = int(freeze_stages)
        self.freeze_mode = freeze_mode
        self.schedule = parse_schedule(unfreeze_schedule)
        self.bn_unfreeze = bn_unfreeze
        self.backbone_lr_mult = float(backbone_lr_mult)
        self.stage_lr_mults = dict(stage_lr_mults or {})
        stages = dict(model.stages())
        if self.stage_lr_mults.keys() - stages.keys():
            raise ValueError("unknown stage learning-rate multiplier")
        if any(not math.isfinite(v) or v < 0 for v in (self.backbone_lr_mult, *self.stage_lr_mults.values())):
            raise ValueError("learning-rate multipliers must be finite and nonnegative")
        if any(n > len(stages) for n in self.schedule.values()):
            raise ValueError("schedule stage count exceeds backbone depth")
        self.load_report = load_report
        self.freeze = self.freezer.stages(self.initial_level, mode=freeze_mode, bn_unfreeze=bn_unfreeze)
        self.history = []
        self.frozen_checks = []
        self._frozen_reference = None
        self.probe = None

    @classmethod
    def build(cls, arch, output_dim, *, pretrained=None, initializer=None, dropout=0., dropout_seed=None,
              conv_backend="auto", input_norm=None, name=None, head="linear", head_hidden=(512, 512),
              head_dropout=.3, feature_adapter=None, spatial_size=None, lora_rank=0, lora_alpha=None, **strategy):
        registry = PretrainedModelRegistry()
        model = registry.build(arch, output_dim, initializer=initializer, dropout=dropout, dropout_seed=dropout_seed,
                               conv_backend=conv_backend, input_norm=input_norm, name=name)
        report = CheckpointLoader().load_converted(model, pretrained) if pretrained is not None else None
        width = int(model.fc_W.value.shape[0])
        if feature_adapter is not None:
            adapter = FeatureAdapter(width, spatial_size=spatial_size or (1, 1), vector=arch == "alexnet",
                                     initializer=initializer, backend=conv_backend, seed=dropout_seed, **feature_adapter)
            model.transfer_adapter = adapter
            model.params.extend(adapter.params)
            model.sub_layers.append(adapter)
            width = adapter.output_dim
        if head == "mlp" or feature_adapter is not None:
            original = {model.fc_W, model.fc_b}
            replacement = HeadAdapterBuilder.build(width, output_dim, kind=head, hidden=head_hidden, dropout=head_dropout,
                                                  initializer=initializer, backend=conv_backend, seed=dropout_seed)
            model.transfer_head = replacement
            model.params = [p for p in model.params if p not in original] + replacement.params
            model.sub_layers.append(replacement)
        elif head not in ("linear", "regression"):
            raise ValueError(f"unknown head: {head}")
        manager = cls(model, load_report=report, **strategy)
        if lora_rank:
            from .lora import attach_resnet_lora

            if manager.schedule or feature_adapter is not None:
                raise ValueError("LoRA placement uses fixed base parameters and the original ResNet features")
            manager.lora = attach_resnet_lora(model,lora_rank,alpha=lora_alpha,seed=dropout_seed)
            manager.freezer = ParameterFreezer(model)
            manager.initial_level = manager.freezer.level = len(model.stages())
            manager.freeze = freeze_summary(model)
        return manager

    def epoch(self, epoch):
        level = self.initial_level
        for boundary, selected in self.schedule.items():
            if boundary <= epoch:
                level = selected
        if level != self.freezer.level:
            self.verify_frozen()
            self.freeze = self.freezer.stages(level, mode=self.freeze_mode, bn_unfreeze=self.bn_unfreeze)
            self.begin_tracking()
        entry = {"epoch": int(epoch), "freeze_stages": level, "trainable_values": self.freeze["trainable_values"]}
        if not self.history or self.history[-1]["epoch"] != epoch:
            self.history.append(entry)
        return entry

    def _frozen_hashes(self):
        params, buffers = named_state(self.model)
        tensors = {name: p.value for name, p in params.items() if not p.trainable}
        tensors.update({name: getattr(layer, attr) for name, (layer, attr) in buffers.items() if layer.frozen})
        result = {}
        for name, value in tensors.items():
            array = asnumpy(value)
            sha = hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode())
            sha.update(array.tobytes())
            result[name] = sha.hexdigest()
        return result

    def begin_tracking(self):
        self._frozen_reference = self._frozen_hashes()

    def verify_frozen(self):
        if self._frozen_reference is None:
            self.begin_tracking()
            return True
        current = self._frozen_hashes()
        changed = [key for key, value in self._frozen_reference.items() if current.get(key) != value]
        check = {"freeze_stages": self.freezer.level, "tensors": len(self._frozen_reference), "unchanged": not changed}
        self.frozen_checks.append(check)
        if changed:
            raise RuntimeError(f"frozen tensors changed: {changed[:5]}")
        return True

    def state_dict(self):
        params, buffers = named_state(self.model)
        return {"level": self.freezer.level, "history": self.history, "checks": self.frozen_checks,
                "frozen_parameters": [name for name, p in params.items() if not p.trainable],
                "bn_frozen": {name: layer.frozen for name, (layer, attr) in buffers.items()},
                "frozen_reference": self._frozen_reference, "probe": self.probe}

    def restore(self, state):
        params, buffers = named_state(self.model)
        frozen = set(state["frozen_parameters"])
        if frozen - params.keys() or set(state["bn_frozen"]) != buffers.keys():
            raise ValueError("freeze state names differ from the current model")
        for name, node in params.items():
            node.trainable = name not in frozen
        for name, (layer, attr) in buffers.items():
            self.freezer._statistics(layer, bool(state["bn_frozen"][name]))
        self.freezer.level = int(state["level"])
        self.freeze = freeze_summary(self.model)
        self.history = list(state["history"])
        self.frozen_checks = list(state["checks"])
        self._frozen_reference = state.get("frozen_reference")
        self.probe = state.get("probe")
        self.verify_frozen()

    def parameter_groups(self):
        groups, assigned = [], set()
        for name, layers in self.model.stages():
            ids = {p for layer in _stage_layers(layers) for p in getattr(layer, "params", ())}
            params = [p for p in self.model.params if p in ids and p not in assigned]
            if params:
                groups.append({"name": name, "params": params, "lr_mult": self.stage_lr_mults.get(name, self.backbone_lr_mult)})
                assigned.update(params)
        head = getattr(self.model, "transfer_head", None)
        head_params = set(head.params) if head else {self.model.fc_W, self.model.fc_b}
        adapter = getattr(self.model, "transfer_adapter", None)
        if adapter:
            head_params.update(adapter.params)
        extra = [p for p in self.model.params if p not in assigned and p not in head_params]
        if extra:
            groups.append({"name": "backbone_extra", "params": extra, "lr_mult": self.backbone_lr_mult})
        groups.append({"name": "head", "params": [p for p in self.model.params if p in head_params], "lr_mult": 1.})
        return groups

    def cache_features(self, directory, splits, *, sample_manifest, input_node, batch, load_input, preprocessing):
        from ..core.graph import Graph
        from ..core.node import Variable
        from ..ops.batchnorm import BatchNorm2d_Op
        from ..ops.dropout import Dropout_Op
        from .cache import FeatureCache

        if self.schedule and any(level != len(self.model.stages()) for level in self.schedule.values()):
            raise ValueError("feature caching requires the backbone to remain frozen for every epoch")
        feature = self.model._last_transfer_features
        graph = Graph(feature)
        if input_node.value.dtype != np.float32:
            raise ValueError("feature caching requires FP32 inputs")
        if any(isinstance(n, Variable) and n.trainable for n in graph.nodes):
            raise ValueError("feature caching requires every pre-head parameter to be frozen")
        if any(isinstance(n, BatchNorm2d_Op) and n.training for n in graph.nodes):
            raise ValueError("feature caching requires frozen BN statistics")
        if any(isinstance(n, Dropout_Op) and n.p for n in graph.nodes):
            raise ValueError("feature caching requires deterministic backbone features, disable backbone dropout")
        params, buffers = named_state(self.model)
        required = set(graph.nodes)
        tensors = {k: p.value for k, p in params.items() if p in required}
        tensors.update({k: getattr(layer, attr) for k, (layer, attr) in buffers.items()
                        if getattr(layer, "_last_op", None) in required})
        sha = hashlib.sha256()
        for key, array in tensors.items():
            sha.update(key.encode())
            sha.update(str(array.dtype).encode())
            sha.update(str(array.shape).encode())
            sha.update(asnumpy(array).tobytes())
        adapter = getattr(self.model, "transfer_adapter", None)
        width = adapter.output_dim if adapter else int(self.model.fc_W.value.shape[0])

        def encode(selected):
            images = np.asarray([load_input(sample) for sample in selected], dtype=np.float32)
            if len(selected) < batch:
                images = np.concatenate((images, np.repeat(images[-1:], batch-len(selected), axis=0)))
            input_node.value = images
            graph.forward()
            return np.ascontiguousarray(asnumpy(feature.value)[:len(selected)])

        return FeatureCache(directory).materialize(splits, sample_manifest=sample_manifest, weights_sha256=sha.hexdigest(),
                                                   preprocessing=preprocessing, batch=batch, width=width, encode=encode)

    def result(self):
        return {"arch": self.model.arch, "freeze_stages": self.freezer.level, "freeze": self.freeze,
                "lora": getattr(self, "lora", None), "probe": self.probe,
                "initial_freeze_stages": self.initial_level, "frozen_checks": list(self.frozen_checks),
                "load_report": summarize_load_report(self.load_report) if self.load_report else None,
                "freeze_history": list(self.history), "backbone_lr_mult": self.backbone_lr_mult,
                "stage_lr_mults": self.stage_lr_mults, "freeze_mode": self.freeze_mode, "bn_unfreeze": self.bn_unfreeze}
