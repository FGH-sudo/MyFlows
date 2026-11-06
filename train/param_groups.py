"""Validated parameter groups with either an absolute LR or a multiplier of the optimizer LR."""

import math
from ..core.node import Variable


def _number(value, field):
    result = float(value)
    if result < 0 or not math.isfinite(result):
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


class ParameterGroups:
    def __init__(self, groups, nodes):
        allowed = set(nodes)
        self.groups = []
        self.by_node = {}
        names = set()
        for index, spec in enumerate(groups or ()):
            name = str(spec.get("name", f"group{index}"))
            if name in names or not name:
                raise ValueError("parameter group names must be unique and nonempty")
            names.add(name)
            if "lr" in spec and "lr_mult" in spec:
                raise ValueError("a parameter group must choose lr or lr_mult")
            params = tuple(spec["params"])
            if not params or len(set(params)) != len(params) or any(not isinstance(p, Variable) or p not in allowed or p in self.by_node for p in params):
                raise ValueError(f"group {name} has empty, duplicate or unknown parameters")
            group = {"name": name, "params": params, "weight_decay": _number(spec.get("weight_decay", 0.), "weight_decay")}
            if "lr" in spec:
                group["lr"] = _number(spec["lr"], "group lr")
            else:
                group["lr_mult"] = _number(spec.get("lr_mult", 1.), "lr_mult")
            self.groups.append(group)
            self.by_node.update((p, group) for p in params)

    @staticmethod
    def _lr(group, base_lr):
        return group["lr"] if "lr" in group else base_lr * group["lr_mult"]

    def options(self, node, base_lr):
        group = self.by_node.get(node)
        return (base_lr, 0.) if group is None else (self._lr(group, base_lr), group["weight_decay"])

    def rates(self, base_lr):
        return {g["name"]: self._lr(g, base_lr) for g in self.groups}

    def serialize(self, key_by_id, base_lr):
        return [{**g, "params": [key_by_id[id(p)] for p in g["params"]], "effective_lr": self._lr(g, base_lr)}
                for g in self.groups]

    @staticmethod
    def restore(records, nodes_by_name):
        specs = []
        for record in records:
            try:
                params = [nodes_by_name[k] for k in record["params"]]
            except KeyError as exc:
                raise ValueError(f"checkpoint group refers to missing parameter {exc}") from exc
            specs.append({**record, "params": params})
        return ParameterGroups(specs, nodes_by_name.values())
