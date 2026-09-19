"""Reduce local train-batch metric stats into a global summary."""

from __future__ import annotations

import math


LEGAL_KINDS = {"mean", "rmse"}


def values_from_stats(stats):
    values = {}
    for name, spec in (stats or {}).items():
        count = int(spec.get("count") or 0)
        if count <= 0:
            continue
        kind = spec.get("kind") or "mean"
        total = float(spec.get("sum") or 0.0)
        if kind == "rmse":
            values[name] = math.sqrt(total / count)
        else:
            values[name] = total / count
    return values


def reduce_step_metrics(rows, n_workers):
    issues = []
    rows = list(rows or [])
    if not rows:
        return None, ["no worker metrics"]
    workers = [row.get("worker_id") for row in rows]
    if len(workers) != len(set(workers)):
        issues.append("duplicate worker metrics")
    expected = set(range(int(n_workers)))
    missing = expected - set(workers)
    extra = set(workers) - expected
    if missing:
        issues.append(f"missing workers {sorted(missing)}")
    if extra:
        issues.append(f"unexpected workers {sorted(extra)}")
    scopes = {row.get("scope") or "train_batch" for row in rows}
    if scopes - {"train_batch"}:
        issues.append(f"unexpected metric scope {sorted(scopes)}")
    aggregations = {row.get("aggregation") or "local" for row in rows}
    if "global" in aggregations and "local" in aggregations:
        issues.append("mixed local and global metric rows")
    epochs = {row.get("epoch") for row in rows}
    steps = {row.get("global_step") for row in rows}
    if len(epochs) != 1:
        issues.append("epoch mismatch")
    if len(steps) != 1:
        issues.append("global_step mismatch")
    if issues:
        return None, issues
    if aggregations == {"global"}:
        if len(rows) != 1:
            return None, ["multiple global metric rows"]
        merged = dict(rows[0])
        merged["aggregation"] = "global"
        merged["scope"] = "train_batch"
        return merged, []
    names = set()
    for row in rows:
        stats = row.get("stats") or {}
        if not stats:
            return None, ["missing local metric stats"]
        if row.get("aggregation") == "global":
            return None, ["refusing to re-aggregate global metrics"]
        names |= set(stats)
    names.add("loss")
    for row in rows:
        stats = row.get("stats") or {}
        absent = names - set(stats)
        if absent:
            issues.append(f"worker {row.get('worker_id')} missing metrics {sorted(absent)}")
        for name, spec in stats.items():
            kind = spec.get("kind") or "mean"
            if kind not in LEGAL_KINDS:
                issues.append(f"illegal kind for {name}: {kind}")
            count = spec.get("count")
            try:
                count_i = int(count)
            except (TypeError, ValueError):
                issues.append(f"illegal count for {name}")
                continue
            if count_i != count or count_i <= 0:
                issues.append(f"illegal count for {name}")
            try:
                total = float(spec.get("sum"))
            except (TypeError, ValueError):
                issues.append(f"non-finite sum for {name}")
                continue
            if not math.isfinite(total):
                issues.append(f"non-finite sum for {name}")
            if name == "accuracy" and count_i != int(row.get("n_samples") or 0):
                issues.append("accuracy count must equal n_samples")
    kinds = {}
    for row in rows:
        for name, spec in (row.get("stats") or {}).items():
            kind = spec.get("kind") or "mean"
            previous = kinds.get(name)
            if previous is None:
                kinds[name] = kind
            elif previous != kind:
                issues.append(f"kind conflict for {name}")
    if issues:
        return None, issues
    merged_stats = {}
    n_samples = 0
    for row in rows:
        n_samples += int(row.get("n_samples") or 0)
        for name, spec in row["stats"].items():
            kind = spec.get("kind") or "mean"
            bucket = merged_stats.setdefault(name, {"sum": 0.0, "count": 0, "kind": kind})
            bucket["sum"] += float(spec.get("sum") or 0.0)
            bucket["count"] += int(spec.get("count") or 0)
    values = values_from_stats(merged_stats)
    merged = {
        "epoch": rows[0].get("epoch", 0),
        "global_step": rows[0].get("global_step", 0),
        "n_samples": n_samples,
        "loss": values.get("loss"),
        "aggregation": "global",
        "scope": "train_batch",
        "stats": merged_stats,
    }
    for name, value in values.items():
        if name != "loss":
            merged[name] = value
    return merged, []
