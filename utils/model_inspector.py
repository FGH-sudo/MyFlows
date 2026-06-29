# -*- coding: utf-8 -*-
"""Lightweight model and graph diagnostics."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from MyFlows.core.device import asnumpy


def _to_numpy(value) -> np.ndarray:
    if value is None:
        return np.asarray([])
    return np.asarray(asnumpy(value))


def model_summary(model) -> dict[str, object]:
    rows = []
    total = 0
    for index, param in enumerate(getattr(model, "params", []) or []):
        value = _to_numpy(getattr(param, "value", None))
        count = int(value.size)
        total += count
        rows.append(
            {
                "index": index,
                "name": getattr(param, "name", f"param_{index}"),
                "shape": tuple(int(v) for v in value.shape),
                "params": count,
                "trainable": bool(getattr(param, "trainable", False)),
            }
        )
    return {"total_params": total, "parameters": rows}


def format_model_summary(summary: Mapping[str, object]) -> str:
    lines = [f"total_params={int(summary.get('total_params', 0))}"]
    for row in summary.get("parameters", []):
        lines.append(f"{row['index']:03d} {row['name']} shape={row['shape']} params={row['params']}")
    return "\n".join(lines)


def inspect_graph(graph, *, check_shape: bool = True, check_content: bool = False) -> dict[str, object]:
    nodes = []
    ok = True
    for index, node in enumerate(getattr(graph, "nodes", []) or []):
        value = _to_numpy(getattr(node, "value", None))
        item = {
            "index": index,
            "name": getattr(node, "name", node.__class__.__name__),
            "op": node.__class__.__name__,
            "shape": tuple(int(v) for v in value.shape) if value.size or value.shape else (),
            "has_nan": False,
            "has_inf": False,
            "all_zero": False,
        }
        if check_shape and getattr(node, "value", None) is None:
            ok = False
            item["missing_value"] = True
        if check_content and value.size:
            item["has_nan"] = bool(np.isnan(value).any())
            item["has_inf"] = bool(np.isinf(value).any())
            item["all_zero"] = bool(np.all(value == 0))
            if item["has_nan"] or item["has_inf"]:
                ok = False
        nodes.append(item)
    return {"ok": ok, "nodes": nodes}


def format_inspection_report(report: Mapping[str, object]) -> str:
    lines = [f"graph_ok={bool(report.get('ok', False))}"]
    for item in report.get("nodes", []):
        flags = []
        if item.get("has_nan"):
            flags.append("nan")
        if item.get("has_inf"):
            flags.append("inf")
        if item.get("all_zero"):
            flags.append("all_zero")
        suffix = f" flags={','.join(flags)}" if flags else ""
        lines.append(f"{item['index']:03d} {item['op']} {item['name']} shape={item['shape']}{suffix}")
    return "\n".join(lines)
