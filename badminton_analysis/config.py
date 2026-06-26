"""YAML configuration loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


@dataclass
class ScreenRegion:
    left: int = 100
    top: int = 100
    width: int = 1280
    height: int = 720


@dataclass
class AppConfig:
    default_model: str = "weights/yolo11n-pose.pt"
    imgsz: int = 960
    conf: float = 0.25
    frame_skip: int = 1
    save_output: bool = False
    screen_region: ScreenRegion = field(default_factory=ScreenRegion)
    outputs_dir: Path = field(default_factory=lambda: ROOT / "outputs")
    runs_dir: Path = field(default_factory=lambda: ROOT / "outputs" / "runs")
    uploads_dir: Path = field(default_factory=lambda: ROOT / "outputs" / "uploads")


# Convenience exports for app.py
OUTPUTS_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "outputs" / "runs"
UPLOADS_DIR = ROOT / "outputs" / "uploads"
VIDEOS_DIR = ROOT / "outputs" / "videos"


def _coerce_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    return Path(str(value))


def load_config(path: Path | None = None) -> AppConfig:
    """Load config from YAML; missing file returns defaults."""
    config_path = path or DEFAULT_CONFIG_PATH
    cfg = AppConfig()
    if not config_path.exists():
        return cfg
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "default_model" in data:
        cfg.default_model = str(data["default_model"])
    if "imgsz" in data:
        cfg.imgsz = int(data["imgsz"])
    if "conf" in data:
        cfg.conf = float(data["conf"])
    if "frame_skip" in data:
        cfg.frame_skip = int(data["frame_skip"])
    if "save_output" in data:
        cfg.save_output = bool(data["save_output"])
    sr = data.get("screen_region", {}) or {}
    cfg.screen_region = ScreenRegion(
        left=int(sr.get("left", cfg.screen_region.left)),
        top=int(sr.get("top", cfg.screen_region.top)),
        width=int(sr.get("width", cfg.screen_region.width)),
        height=int(sr.get("height", cfg.screen_region.height)),
    )
    if "outputs_dir" in data:
        cfg.outputs_dir = _coerce_path(data["outputs_dir"])
    if "runs_dir" in data:
        cfg.runs_dir = _coerce_path(data["runs_dir"])
    if "uploads_dir" in data:
        cfg.uploads_dir = _coerce_path(data["uploads_dir"])
    return cfg
