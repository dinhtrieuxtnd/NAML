from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictModel):
    name: str
    output_root: Path


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return config


def save_config(config: dict, config_path: str | Path) -> None:
    path = Path(config_path)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, default_flow_style=False, allow_unicode=True)
