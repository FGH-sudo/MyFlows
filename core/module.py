# -*- coding: utf-8 -*-
"""自动注册参数、缓冲区与子模块的 Module 体系（向后兼容 legacy Layer）。"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .device import asnumpy, xp
from .node import Variable


class Parameter(Variable):
    """可自动注册到 :class:`Module` 的参数节点（默认 ``trainable=True``）。"""

    def __init__(
        self,
        data=None,
        *,
        trainable: bool = True,
        requires_grad: bool | None = None,
        name: str | None = None,
    ) -> None:
        if requires_grad is not None and bool(requires_grad) != bool(trainable):
            raise ValueError("use trainable to freeze or unfreeze a Parameter")
        super().__init__(
            data,
            trainable=bool(trainable),
            name=name,
            constant=False,
            requires_grad=bool(trainable),
        )
        self._is_parameter = True


class Buffer:
    """可自动注册到 :class:`Module` 的持久/非持久张量缓冲区（如 BN 统计量）。"""

    def __init__(self, data, *, persistent: bool = True, name: str | None = None) -> None:
        if data is None or not hasattr(data, "shape"):
            raise TypeError("Buffer 需要带 .shape 属性的 NumPy 或 CuPy 数组")
        self.data = data
        self.persistent = bool(persistent)
        self.name = name

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def value(self):
        return self.data

    @value.setter
    def value(self, arr) -> None:
        self.data = arr

    def copy_from(self, value) -> None:
        arr = xp.asarray(value, dtype=self.data.dtype)
        if hasattr(self.data, "shape") and arr.shape == self.data.shape:
            self.data[...] = arr
        else:
            self.data = arr


def _is_legacy_layer(value) -> bool:
    return (
        value is not None
        and not isinstance(value, (Module, Variable, Buffer))
        and hasattr(value, "params")
        and hasattr(value, "forward")
    )


class Module:
    """支持自动追踪子模块、``Parameter`` 与 ``Buffer`` 的神经网络模块基类。"""

    def __init__(self, name: str | None = None) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)
        object.__setattr__(self, "name", name)

    def __setattr__(self, name: str, value) -> None:
        params = self.__dict__.get("_parameters")
        buffers = self.__dict__.get("_buffers")
        modules = self.__dict__.get("_modules")
        if params is None or buffers is None or modules is None:
            object.__setattr__(self, name, value)
            return

        if isinstance(value, Parameter):
            buffers.pop(name, None)
            modules.pop(name, None)
            if getattr(value, "name", None) in (None, "", "Variable"):
                value.name = name
            params[name] = value
        elif isinstance(value, Buffer):
            params.pop(name, None)
            modules.pop(name, None)
            if value.name is None:
                value.name = name
            buffers[name] = value
        elif isinstance(value, Module) or _is_legacy_layer(value):
            params.pop(name, None)
            buffers.pop(name, None)
            modules[name] = value
        elif value is None:
            params.pop(name, None)
            buffers.pop(name, None)
            modules.pop(name, None)
        object.__setattr__(self, name, value)

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        if not name or "." in name:
            raise ValueError(f"非法参数名: {name!r}")
        if param is not None and not isinstance(param, Parameter):
            raise TypeError(f"register_parameter 需要 Parameter 或 None，收到 {type(param).__name__}")
        setattr(self, name, param)

    def register_buffer(self, name: str, data, *, persistent: bool = True) -> None:
        if not name or "." in name:
            raise ValueError(f"非法缓冲区名: {name!r}")
        if data is None:
            setattr(self, name, None)
            return
        buf = data if isinstance(data, Buffer) else Buffer(data, persistent=persistent, name=name)
        buf.persistent = bool(persistent)
        setattr(self, name, buf)

    def add_module(self, name: str, module) -> None:
        if not name or "." in name:
            raise ValueError(f"非法子模块名: {name!r}")
        if module is not None and not (isinstance(module, Module) or _is_legacy_layer(module)):
            raise TypeError(f"add_module 需要 Module 或 Layer，收到 {type(module).__name__}")
        setattr(self, name, module)

    def _named_legacy_params(self, layer, prefix: str, seen: set[int]) -> Iterator[tuple[str, Variable]]:
        for idx, p in enumerate(getattr(layer, "params", [])):
            if id(p) in seen:
                continue
            seen.add(id(p))
            pname = getattr(p, "name", None) or f"param_{idx}"
            yield f"{prefix}{pname}", p

    def _named_legacy_buffers(self, layer, prefix: str, seen: set[int]) -> Iterator[tuple[str, object]]:
        stack = [layer]
        visited_layers = set()
        index = 0
        while stack:
            cur = stack.pop(0)
            if id(cur) in visited_layers:
                continue
            visited_layers.add(id(cur))
            for attr in ("running_mean", "running_var"):
                arr = getattr(cur, attr, None)
                if arr is not None and hasattr(arr, "shape") and id(arr) not in seen:
                    seen.add(id(arr))
                    yield f"{prefix}{index}.{attr}" if index else f"{prefix}{attr}", arr
            index += 1
            for sub in getattr(cur, "sub_layers", []):
                stack.append(sub)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        *,
        _seen_params: set[int] | None = None,
        _seen_modules: set[int] | None = None,
    ) -> Iterator[tuple[str, Variable]]:
        seen_p = set() if _seen_params is None else _seen_params
        seen_m = set() if _seen_modules is None else _seen_modules
        if id(self) in seen_m:
            return
        seen_m.add(id(self))

        for name, param in self._parameters.items():
            if param is not None and id(param) not in seen_p:
                seen_p.add(id(param))
                yield f"{prefix}{name}", param

        if not recurse:
            return
        for name, mod in self._modules.items():
            if mod is None:
                continue
            sub_prefix = f"{prefix}{name}."
            if isinstance(mod, Module):
                yield from mod.named_parameters(
                    sub_prefix,
                    recurse=True,
                    _seen_params=seen_p,
                    _seen_modules=seen_m,
                )
            else:
                yield from self._named_legacy_params(mod, sub_prefix, seen_p)

    def parameters(self, recurse: bool = True) -> Iterator[Variable]:
        for _name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_buffers(
        self,
        prefix: str = "",
        recurse: bool = True,
        *,
        persistent_only: bool = False,
        _seen_bufs: set[int] | None = None,
        _seen_modules: set[int] | None = None,
    ) -> Iterator[tuple[str, object]]:
        seen_b = set() if _seen_bufs is None else _seen_bufs
        seen_m = set() if _seen_modules is None else _seen_modules
        if id(self) in seen_m:
            return
        seen_m.add(id(self))

        for name, buf in self._buffers.items():
            if buf is None:
                continue
            if persistent_only and not getattr(buf, "persistent", True):
                continue
            if id(buf) not in seen_b:
                seen_b.add(id(buf))
                yield f"{prefix}{name}", buf

        if not recurse:
            return
        for name, mod in self._modules.items():
            if mod is None:
                continue
            sub_prefix = f"{prefix}{name}."
            if isinstance(mod, Module):
                yield from mod.named_buffers(
                    sub_prefix,
                    recurse=True,
                    persistent_only=persistent_only,
                    _seen_bufs=seen_b,
                    _seen_modules=seen_m,
                )
            else:
                yield from self._named_legacy_buffers(mod, sub_prefix, seen_b)

    def buffers(self, recurse: bool = True, *, persistent_only: bool = False) -> Iterator[object]:
        for _name, buf in self.named_buffers(recurse=recurse, persistent_only=persistent_only):
            yield buf

    def children(self) -> Iterator[object]:
        seen = set()
        for mod in self._modules.values():
            if mod is not None and id(mod) not in seen:
                seen.add(id(mod))
                yield mod

    def named_modules(self, prefix: str = "", _seen: set[int] | None = None) -> Iterator[tuple[str, object]]:
        seen = set() if _seen is None else _seen
        if id(self) in seen:
            return
        seen.add(id(self))
        yield prefix, self
        for name, mod in self._modules.items():
            if mod is None or id(mod) in seen:
                continue
            sub_prefix = f"{prefix}.{name}" if prefix else name
            if isinstance(mod, Module):
                yield from mod.named_modules(sub_prefix, _seen=seen)
            else:
                seen.add(id(mod))
                yield sub_prefix, mod

    def modules(self) -> Iterator[object]:
        for _name, mod in self.named_modules():
            yield mod

    def train(self, mode: bool = True, *, _seen: set[int] | None = None) -> "Module":
        seen = set() if _seen is None else _seen
        if id(self) in seen:
            return self
        seen.add(id(self))
        object.__setattr__(self, "training", bool(mode))
        for child in self.children():
            if id(child) in seen:
                continue
            if isinstance(child, Module):
                child.train(mode, _seen=seen)
            elif hasattr(child, "train"):
                seen.add(id(child))
                child.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} 必须实现 forward()")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def params(self) -> list[Variable]:
        return list(self.parameters(recurse=True))

    @property
    def sub_layers(self) -> list[object]:
        return [m for m in self._modules.values() if m is not None]

    def state_dict(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, param in self.named_parameters(recurse=True):
            out[name] = np.array(asnumpy(param.value), copy=True)
        for name, buf in self.named_buffers(recurse=True, persistent_only=True):
            arr = buf.data if isinstance(buf, Buffer) else buf
            out[name] = np.array(asnumpy(arr), copy=True)
        return out

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> dict[str, list[str]]:
        if not isinstance(state, dict):
            raise TypeError("state_dict 必须是 dict")
        param_map = dict(self.named_parameters(recurse=True))
        buf_map = dict(self.named_buffers(recurse=True, persistent_only=True))
        expected = set(param_map) | set(buf_map)
        provided = set(state)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if strict and (missing or unexpected):
            parts = []
            if missing:
                parts.append(f"缺少键: {', '.join(missing[:8])}")
            if unexpected:
                parts.append(f"多余键: {', '.join(unexpected[:8])}")
            raise ValueError("state_dict 键不匹配: " + "; ".join(parts))

        staged: dict[str, object] = {}
        for key in sorted(expected & provided):
            raw = np.asarray(state[key])
            if key in param_map:
                target = param_map[key].value
                if target is not None and raw.shape != tuple(target.shape):
                    raise ValueError(f"参数 {key} 形状不匹配: 期望 {target.shape}，收到 {raw.shape}")
                dtype = target.dtype if target is not None else raw.dtype
                staged[key] = xp.asarray(raw, dtype=dtype).copy()
            else:
                buf = buf_map[key]
                cur = buf.data if isinstance(buf, Buffer) else buf
                if raw.shape != tuple(cur.shape):
                    raise ValueError(f"缓冲区 {key} 形状不匹配: 期望 {cur.shape}，收到 {raw.shape}")
                staged[key] = raw

        for key, value in staged.items():
            if key in param_map:
                param_map[key].value = value
            else:
                buf = buf_map[key]
                if isinstance(buf, Buffer):
                    buf.copy_from(value)
                else:
                    buf[...] = xp.asarray(value, dtype=buf.dtype)

        return {"missing": missing, "unexpected": unexpected}
