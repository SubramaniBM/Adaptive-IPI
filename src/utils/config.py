"""
src/utils/config.py

YAML configuration loading and merging for the Adaptive-IPI project.

Provides a simple config loader that reads YAML files and returns
plain dictionaries. Supports loading multiple config files and
merging them (later files override earlier ones).
"""

from pathlib import Path
from typing import Any, Optional, Union

import yaml


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load a single YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    return config


def load_configs(*paths: Union[str, Path]) -> dict[str, Any]:
    """Load and merge multiple YAML configuration files.

    Later files override values from earlier files. Nested dicts
    are merged recursively; non-dict values are replaced.

    Args:
        *paths: Paths to YAML files, in order of increasing priority.

    Returns:
        Merged configuration dictionary.
    """
    merged: dict[str, Any] = {}
    for path in paths:
        config = load_config(path)
        merged = _deep_merge(merged, config)
    return merged


def save_config(config: dict[str, Any], path: Union[str, Path]) -> None:
    """Save a configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_nested(
    config: dict[str, Any],
    *keys: str,
    default: Optional[Any] = None,
) -> Any:
    """Safely retrieve a nested value from a config dictionary.

    Args:
        config: Configuration dictionary.
        *keys: Sequence of keys to traverse.
        default: Value to return if the key path does not exist.

    Returns:
        The value at the specified key path, or default.

    Example:
        >>> cfg = {"training": {"lr": 1e-5}}
        >>> get_nested(cfg, "training", "lr")
        1e-05
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries.

    Values in ``override`` take precedence. Nested dicts are merged
    recursively; all other types are replaced outright.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
