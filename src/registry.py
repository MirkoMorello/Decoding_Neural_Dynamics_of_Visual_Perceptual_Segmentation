# src/registry.py
"""
Central registry for model and data builders.

This file contains the registry dictionaries and the decorator functions.
It has no other internal project dependencies, so it can be safely imported
by any builder file without causing circular imports.
"""
from typing import Any, Dict, Callable

MODEL_REGISTRY: Dict[str, Any] = {}
DATA_REGISTRY: Dict[str, Any] = {}

def register_model(key: str):
    """Decorator to register a model builder function."""
    def inner(fn: Callable) -> Callable:
        if key in MODEL_REGISTRY:
            raise ValueError(f"Model key '{key}' already registered")
        MODEL_REGISTRY[key] = fn
        return fn
    return inner

def register_data(key: str):
    """Decorator to register a dataset builder function."""
    def inner(fn: Callable) -> Callable:
        if key in DATA_REGISTRY:
            raise ValueError(f"Dataset key '{key}' already registered")
        DATA_REGISTRY[key] = fn
        return fn
    return inner