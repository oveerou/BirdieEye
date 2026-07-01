"""SQLite history and metrics.json writer."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class RunMetrics:
    run_id: str
    source_type: str
    source_ref: str
    model_path: str
    device: str
    conf: float
    imgsz: int
    frame_skip: int
    total_frames: int = 0
    avg_fps: float = 0.0
    avg_player_count: float = 0.0
    ball_visible_ratio: float = 0.0
    upper_avg_speed: float = 0.0
    lower_avg_speed: float = 0.0
    total_rallies: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_ref TEXT,
    model_path TEXT,
    device TEXT,
    conf REAL,
    imgsz INTEGER,
    frame_skip INTEGER,
    total_frames INTEGER,
    avg_fps REAL,
    avg_player_count REAL,
    ball_visible_ratio REAL,
    upper_avg_speed REAL,
    lower_avg_speed REAL,
    total_rallies INTEGER,
    created_at TEXT
);
"""


def init_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (or create) the SQLite DB and return a connection."""
    if db_path is None:
        from .config import OUTPUTS_DIR
        db_path = OUTPUTS_DIR / "badminton.db"
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def insert_run(conn: sqlite3.Connection, m: RunMetrics) -> None:
    d = m.to_dict()
    cols = ", ".join(d.keys())
    placeholders = ", ".join(["?"] * len(d))
    conn.execute(
        f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({placeholders})",
        list(d.values()),
    )
    conn.commit()


def list_runs(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    cur = conn.execute(
        "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def save_metrics_json(m: RunMetrics, run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "metrics.json"
    out.write_text(json.dumps(m.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out
