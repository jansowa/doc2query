"""Loading and validation for composed YAML run configurations."""

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from doc2query.schemas import AppConfig


def load_config(path: Path) -> AppConfig:
    """Load a YAML config, resolve interpolation, and validate its full contract."""
    if not path.is_file():
        raise ValueError(f"configuration file does not exist: {path}")
    try:
        raw = OmegaConf.load(path)
        if "defaults" in raw:
            config_root = next(
                (parent for parent in (path.parent, *path.parents) if parent.name == "configs"),
                None,
            )
            if config_root is None:
                raise ValueError("Hydra config with defaults must be located below configs/")
            relative_name = path.resolve().relative_to(config_root.resolve()).with_suffix("")
            with initialize_config_dir(version_base=None, config_dir=str(config_root.resolve())):
                raw = compose(config_name=relative_name.as_posix())
        resolved: Any = OmegaConf.to_container(raw, resolve=True)
    except Exception as exc:
        raise ValueError(f"cannot load configuration {path}: {exc}") from exc
    if not isinstance(resolved, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return AppConfig.model_validate(resolved)
