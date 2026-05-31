# -*- coding: utf-8 -*-
"""ONNX 模型 INT8 量化工具。"""

from __future__ import annotations

from pathlib import Path


def quantize_onnx_dynamic(
    model_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """
    对 ONNX 模型做动态 INT8 量化（权重量化，激活仍为 float）。
    需要: onnx, onnxruntime
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path(model_path)
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = Path(output_path) if output_path else src.with_name(src.stem + "_int8.onnx")
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
    )
    return dst


def quantize_onnx_static(
    model_path: str | Path,
    output_path: str | Path,
    calibration_reader,
) -> Path:
    """静态量化（需 CalibrationDataReader）。"""
    from onnxruntime.quantization import QuantType, quantize_static

    src = Path(model_path)
    dst = Path(output_path)
    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=calibration_reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    return dst
