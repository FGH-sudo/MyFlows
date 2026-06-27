# -*- coding: utf-8 -*-
"""训练观测指标子模块。"""

from MyFlows.utils.observers.activations import activation_stats, feature_grid_images
from MyFlows.utils.observers.formatting import safe_name, training_config_text
from MyFlows.utils.observers.labels import label_stats_classification, label_stats_regression
from MyFlows.utils.observers.params import (
    array_norm,
    global_gradient_norm,
    gradient_norms,
    named_parameters,
    parameter_norms,
    update_ratios,
)
from MyFlows.utils.observers.tensorboard_writers import log_gradient_histograms, log_parameter_histograms

__all__ = [
    "activation_stats",
    "array_norm",
    "feature_grid_images",
    "global_gradient_norm",
    "gradient_norms",
    "label_stats_classification",
    "label_stats_regression",
    "log_gradient_histograms",
    "log_parameter_histograms",
    "named_parameters",
    "parameter_norms",
    "safe_name",
    "training_config_text",
    "update_ratios",
]
