# -*- coding: utf-8 -*-
"""训练观测文本与 tag 格式化。"""

from __future__ import annotations

import json
from typing import Mapping


def safe_name(name: str | None, fallback: str) -> str:
    raw = str(name or fallback)
    return raw.replace("/", "_").replace("\\", "_").replace(" ", "_")


def training_config_text(args, extra: Mapping[str, object] | None = None) -> str:
    data = dict(vars(args)) if hasattr(args, "__dict__") else {}
    if extra:
        data.update(dict(extra))
    return "```json\n" + json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n```"
