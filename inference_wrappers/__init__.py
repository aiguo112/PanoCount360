"""Thin wrappers around `models.model_factory.build_model` for the analyzer GUI."""

from inference_wrappers.base_model import (
    MODEL_CONFIGS,
    build_and_load,
    run_forward,
)

__all__ = ["MODEL_CONFIGS", "build_and_load", "run_forward"]
