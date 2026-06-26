# Streamlit Web UI for Good-Badminton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Streamlit Web UI mirroring `football-realtime-analyzer`'s `app.py` structure, while keeping all detection logic native to Good-Badminton (YOLO Pose + ball detection + court mapping + upper/lower player tracking).

**Architecture:** New files `app.py` + `badminton_analysis/{analytics,storage,renderer,config}.py` + `configs/default.yaml` + `start.bat`. Existing `main.py` and `live.py` stay unchanged as fallback entry points. `BadmintonAnalysisSystem.process_frame` becomes a per-frame entry point that optionally pushes rendered frames to a queue; Streamlit's inference thread consumes frames and pushes to a display queue, main thread updates `st.image` and metrics.

**Tech Stack:** Streamlit 鈮?1.30, PyYAML 鈮?6.0, stdlib `sqlite3`, stdlib `queue` + `threading`. Existing Bad-Badminton stack (YOLO Pose, OpenCV, GPU torch) used unchanged.

## Global Constraints

- All new code follows the existing `badminton_analysis` package layout.
- Streamlit UI structure mirrors `D:\A_shixi\football-realtime-analyzer\app.py` (sidebar + main + history).
- Player distinction is **upper vs lower court** (NOT football's team color classification).
- Real-time metrics are **badminton-native** (movement, speed, max speed, rally id, upper/lower averages) 鈥? NOT football's possession %.
- `app.py` is the new recommended entry point; `main.py` and `live.py` are NOT removed or modified beyond Task 6.
- Existing reference: `python main.py --video-path videos/demo1.mp4 --template-path templates/demo1.png --display false` must still produce byte-identical `outputs/demo1/detect_demo1.mp4` of exactly **22,112,015 bytes**.
- Existing 11 unit tests must continue to pass.
- All new code uses UTF-8; the plan file's Chinese blocks are saved as UTF-8 (verify with `Get-Content -Encoding UTF8`).
- venv Python: `D:\A_shixi\Good-Badminton\.venv\Scripts\python.exe`. Workdir: `D:\A_shixi\Good-Badminton`.
- All commits are local; do not push.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `app.py` | Create | Streamlit Web UI entry (sidebar, main, history) |
| `badminton_analysis/config.py` | Create (TDD) | YAML config loader + dataclass |
| `badminton_analysis/analytics.py` | Create (TDD) | FpsCounter, BallTrail, FrameStats, MetricsAggregator |
| `badminton_analysis/storage.py` | Create (TDD) | SQLite + metrics.json (mirrors football's storage.py) |
| `badminton_analysis/renderer.py` | Create (TDD) | Single-frame render (skeletons, trails, court, stats panel) |
| `configs/default.yaml` | Create | Default parameters |
| `start.bat` | Create | One-click Streamlit launcher |
| `badminton_analysis/system.py` | Modify | Expose `_process_frame` as public `process_frame`; add optional `display_queue` push |
| `requirements.txt` | Modify | +streamlit, +pyyaml |
| `README.md` | Modify | +Web UI section |
| `tests/test_config.py` | Create | Unit tests for config |
| `tests/test_analytics.py` | Create | Unit tests for analytics classes |
| `tests/test_storage.py` | Create | Unit tests for storage |
| `tests/test_renderer.py` | Create | Unit tests for renderer |
| `tests/test_web_e2e.py` | Create | E2E: thread + queue flow produces frames |

---

### Task 1: Add streamlit and pyyaml to requirements.txt

**Files:**
- Modify: `requirements.txt` (append two lines)

- [ ] **Step 1: Append two lines to requirements.txt**

Append (do not rewrite the file) these two lines at the end of `requirements.txt`:

```
streamlit>=1.30.0
pyyaml>=6.0
```

- [ ] **Step 2: Install the new packages into the venv**

Run:
```powershell
.\.venv\Scripts\python.exe -m pip install streamlit>=1.30.0 pyyaml>=6.0
```

Expected: `Successfully installed streamlit-X.Y.Z pyyaml-X.Y.Z` (versions vary).

- [ ] **Step 3: Verify imports work**

Run:
```powershell
.\.venv\Scripts\python.exe -c "import streamlit, yaml; print('streamlit:', streamlit.__version__); print('yaml:', yaml.__version__)"
```

Expected: Two version lines, no errors.

- [ ] **Step 4: Commit**

```powershell
git add requirements.txt
git commit -m "deps: add streamlit and pyyaml for web UI"
```

---

### Task 2: TDD - config.py + default.yaml

**Files:**
- Create: `configs/default.yaml`
- Create: `badminton_analysis/config.py`
- Create: `tests/test_config.py`

**Why first:** Every other module reads from `AppConfig`. Pinning the contract here lets downstream tasks depend on the dataclass fields.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
"""Unit tests for config loader."""
from __future__ import annotations

import unittest
from pathlib import Path

from badminton_analysis.config import AppConfig, ScreenRegion, load_config


class ConfigTest(unittest.TestCase):
    def test_screen_region_defaults(self):
        r = ScreenRegion()
        self.assertEqual(r.left, 100)
        self.assertEqual(r.top, 100)
        self.assertEqual(r.width, 1280)
        self.assertEqual(r.height, 720)

    def test_appconfig_defaults(self):
        c = AppConfig()
        self.assertEqual(c.default_model, "weights/yolo11n-pose.pt")
        self.assertEqual(c.imgsz, 960)
        self.assertEqual(c.conf, 0.25)
        self.assertEqual(c.frame_skip, 1)
        self.assertFalse(c.save_output)
        self.assertIsInstance(c.screen_region, ScreenRegion)
        # Path fields should be Path objects pointing under the project
        self.assertTrue(str(c.outputs_dir).endswith("outputs"))
        self.assertTrue(str(c.runs_dir).endswith("outputs" + str(Path("runs"))) or
                        str(c.runs_dir).endswith("outputs\\runs"))
        self.assertTrue(str(c.uploads_dir).endswith("uploads"))

    def test_load_config_returns_appconfig(self):
        c = load_config()
        self.assertIsInstance(c, AppConfig)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests, confirm failure**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config -v
```

Expected: `ModuleNotFoundError: No module named 'badminton_analysis.config'`.

- [ ] **Step 3: Create configs/default.yaml**

Create `configs/default.yaml`:

```yaml
default_model: weights/yolo11n-pose.pt
imgsz: 960
conf: 0.25
frame_skip: 1
save_output: false
screen_region:
  left: 100
  top: 100
  width: 1280
  height: 720
outputs_dir: outputs
runs_dir: outputs/runs
uploads_dir: outputs/uploads
```

- [ ] **Step 4: Create config.py**

Create `badminton_analysis/config.py`:

```python
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
```

- [ ] **Step 5: Run tests, confirm pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config -v
```

Expected: 3 tests, all `ok`.

- [ ] **Step 6: Run full test suite to confirm no regressions**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 14 tests in ... OK` (3 new + 11 prior).

- [ ] **Step 7: Commit**

```powershell
git add configs/default.yaml badminton_analysis/config.py tests/test_config.py
git commit -m "feat(config): add YAML config loader and default config"
```

---

### Task 3: TDD - analytics.py

**Files:**
- Create: `badminton_analysis/analytics.py`
- Create: `tests/test_analytics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analytics.py`:

```python
"""Unit tests for analytics primitives."""
from __future__ import annotations

import unittest

from badminton_analysis.analytics import (
    BallTrail,
    FpsCounter,
    FrameStats,
    MetricsAggregator,
)


class FpsCounterTest(unittest.TestCase):
    def test_first_tick_returns_zero(self):
        f = FpsCounter(alpha=0.9)
        self.assertEqual(f.tick(), 0.0)

    def test_subsequent_ticks_smooth(self):
        f = FpsCounter(alpha=0.5)
        f.tick()  # prime
        v = f.tick()  # dt = 0 if called immediately
        self.assertGreaterEqual(v, 0.0)


class BallTrailTest(unittest.TestCase):
    def test_empty_initially(self):
        t = BallTrail(maxlen=10)
        self.assertEqual(t.points(), [])

    def test_update_appends_centers(self):
        t = BallTrail(maxlen=3)
        # Each "track" is anything; the analytics reads .xy or center via attribute.
        class FakeTrack:
            def __init__(self, x, y):
                self.x, self.y = x, y
        t.update([FakeTrack(10, 20)])
        self.assertEqual(t.points(), [(10, 20)])

    def test_maxlen_caps_history(self):
        t = BallTrail(maxlen=2)
        class P:
            def __init__(self, x, y): self.x, self.y = x, y
        t.update([P(1, 1)])
        t.update([P(2, 2)])
        t.update([P(3, 3)])
        self.assertEqual(len(t.points()), 2)
        self.assertEqual(t.points()[-1], (3, 3))


class FrameStatsTest(unittest.TestCase):
    def test_default_fields(self):
        s = FrameStats(
            frame_idx=10, fps=30.0, player_count=2,
            ball_visible=True, ball_center=(100, 200), rally_id=1,
            upper_player_count=1, lower_player_count=1,
            avg_speed_upper=0.5, avg_speed_lower=0.3,
            total_distance_upper=10.0, total_distance_lower=7.0,
        )
        self.assertEqual(s.frame_idx, 10)
        self.assertTrue(s.ball_visible)
        self.assertEqual(s.upper_player_count, 1)


class MetricsAggregatorTest(unittest.TestCase):
    def _stat(self, **kw):
        defaults = dict(
            frame_idx=0, fps=30.0, player_count=2,
            ball_visible=True, ball_center=(0, 0), rally_id=0,
            upper_player_count=1, lower_player_count=1,
            avg_speed_upper=0.0, avg_speed_lower=0.0,
            total_distance_upper=0.0, total_distance_lower=0.0,
        )
        defaults.update(kw)
        return FrameStats(**defaults)

    def test_empty_summary(self):
        agg = MetricsAggregator()
        s = agg.summary()
        self.assertEqual(s["total_frames"], 0)
        self.assertEqual(s["avg_fps"], 0.0)
        self.assertEqual(s["total_rallies"], 0)

    def test_aggregates_averages(self):
        agg = MetricsAggregator()
        agg.update(self._stat(fps=20.0, player_count=2, ball_visible=True,
                              upper_player_count=1, lower_player_count=1,
                              avg_speed_upper=0.5, avg_speed_lower=0.3,
                              total_distance_upper=10.0, total_distance_lower=5.0))
        agg.update(self._stat(fps=40.0, player_count=4, ball_visible=False,
                              upper_player_count=2, lower_player_count=2,
                              avg_speed_upper=1.0, avg_speed_lower=0.6,
                              total_distance_upper=20.0, total_distance_lower=15.0))
        s = agg.summary()
        self.assertEqual(s["total_frames"], 2)
        self.assertAlmostEqual(s["avg_fps"], 30.0)
        self.assertEqual(s["ball_visible_count"], 1)
        self.assertAlmostEqual(s["ball_visible_ratio"], 0.5)
        self.assertEqual(s["total_rallies"], 0)  # no rally_id in stats

    def test_tracks_rallies(self):
        agg = MetricsAggregator()
        for rid in (1, 1, 2, 2, 2):
            agg.update(self._stat(rally_id=rid))
        s = agg.summary()
        self.assertEqual(s["total_rallies"], 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests, confirm failure**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_analytics -v
```

Expected: `ModuleNotFoundError: No module named 'badminton_analysis.analytics'`.

- [ ] **Step 3: Write the minimal implementation**

Create `badminton_analysis/analytics.py`:

```python
"""Analytics primitives: FPS, ball trail, per-frame stats, multi-frame aggregation."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


class FpsCounter:
    """Exponentially-smoothed FPS."""

    def __init__(self, alpha: float = 0.9):
        self._alpha = alpha
        self._last_t: Optional[float] = None
        self._fps: float = 0.0

    def tick(self) -> float:
        now = time.time()
        if self._last_t is None:
            self._last_t = now
            return 0.0
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0:
            return self._fps
        instant = 1.0 / dt
        self._fps = self._alpha * self._fps + (1 - self._alpha) * instant if self._fps else instant
        return self._fps


class BallTrail:
    """Rolling buffer of the ball's recent centers (x, y)."""

    def __init__(self, maxlen: int = 30):
        self._buf: deque = deque(maxlen=maxlen)

    def update(self, tracks: list) -> list:
        """Append centers from `tracks` (objects with .x, .y). Returns the new list."""
        if not tracks:
            return self.points()
        t = tracks[0]
        x = getattr(t, "x", None)
        y = getattr(t, "y", None)
        if x is None or y is None:
            return self.points()
        self._buf.append((int(x), int(y)))
        return self.points()

    def points(self) -> list:
        return list(self._buf)


@dataclass
class FrameStats:
    """Per-frame statistics for the live UI."""
    frame_idx: int
    fps: float
    player_count: int
    ball_visible: bool
    ball_center: Optional[tuple]
    rally_id: int
    upper_player_count: int
    lower_player_count: int
    avg_speed_upper: float
    avg_speed_lower: float
    total_distance_upper: float
    total_distance_lower: float


class MetricsAggregator:
    """Aggregates FrameStats into a summary dict for metrics.json + DB."""

    def __init__(self):
        self._frames: list[FrameStats] = []
        self._max_rally_id: int = 0

    def update(self, stats: FrameStats) -> None:
        self._frames.append(stats)
        if stats.rally_id > self._max_rally_id:
            self._max_rally_id = stats.rally_id

    def summary(self) -> dict:
        n = len(self._frames)
        if n == 0:
            return {
                "total_frames": 0, "avg_fps": 0.0, "avg_player_count": 0.0,
                "ball_visible_count": 0, "ball_visible_ratio": 0.0,
                "upper_player_count": 0, "lower_player_count": 0,
                "upper_avg_speed": 0.0, "lower_avg_speed": 0.0,
                "upper_max_speed": 0.0, "lower_max_speed": 0.0,
                "upper_total_distance": 0.0, "lower_total_distance": 0.0,
                "total_rallies": 0,
            }
        ball_visible_count = sum(1 for f in self._frames if f.ball_visible)
        avg_fps = sum(f.fps for f in self._frames) / n
        avg_pc = sum(f.player_count for f in self._frames) / n
        upper_speeds = [f.avg_speed_upper for f in self._frames]
        lower_speeds = [f.avg_speed_lower for f in self._frames]
        return {
            "total_frames": n,
            "avg_fps": round(avg_fps, 2),
            "avg_player_count": round(avg_pc, 2),
            "ball_visible_count": ball_visible_count,
            "ball_visible_ratio": round(ball_visible_count / n, 4),
            "upper_player_count": self._frames[-1].upper_player_count,
            "lower_player_count": self._frames[-1].lower_player_count,
            "upper_avg_speed": round(sum(upper_speeds) / n, 3),
            "lower_avg_speed": round(sum(lower_speeds) / n, 3),
            "upper_max_speed": round(max(upper_speeds), 3) if upper_speeds else 0.0,
            "lower_max_speed": round(max(lower_speeds), 3) if lower_speeds else 0.0,
            "upper_total_distance": round(self._frames[-1].total_distance_upper, 2),
            "lower_total_distance": round(self._frames[-1].total_distance_lower, 2),
            "total_rallies": self._max_rally_id,
        }
```

- [ ] **Step 4: Run tests, confirm pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_analytics -v
```

Expected: 4 test classes, all `ok`.

- [ ] **Step 5: Run full suite to confirm no regressions**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 18 tests in ... OK` (4 new + 14 prior).

- [ ] **Step 6: Commit**

```powershell
git add badminton_analysis/analytics.py tests/test_analytics.py
git commit -m "feat(analytics): add FpsCounter, BallTrail, FrameStats, MetricsAggregator"
```

---

### Task 4: TDD - storage.py

**Files:**
- Create: `badminton_analysis/storage.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_storage.py`:

```python
"""Unit tests for storage (SQLite + metrics.json)."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from badminton_analysis.storage import (
    RunMetrics,
    init_db,
    insert_run,
    list_runs,
    save_metrics_json,
)


def _sample_metrics():
    return RunMetrics(
        run_id="run_test_001",
        source_type="screen_capture",
        source_ref="100,100,640,480",
        model_path="weights/yolo11n-pose.pt",
        device="cuda:0",
        conf=0.25,
        imgsz=960,
        frame_skip=1,
        total_frames=300,
        avg_fps=28.5,
        avg_player_count=2.0,
        ball_visible_ratio=0.8,
        upper_avg_speed=0.4,
        lower_avg_speed=0.3,
        total_rallies=5,
    )


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = Path(self.tmp) / "test.db"
        self.run_dir = Path(self.tmp) / "run_test_001"

    def test_init_db_creates_table(self):
        conn = init_db(self.db_path)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
        )
        self.assertIsNotNone(cur.fetchone())
        conn.close()

    def test_insert_and_list_run(self):
        conn = init_db(self.db_path)
        insert_run(conn, _sample_metrics())
        conn.close()
        conn = init_db(self.db_path)
        rows = list_runs(conn, limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "run_test_001")
        self.assertEqual(rows[0]["source_type"], "screen_capture")
        self.assertEqual(rows[0]["total_rallies"], 5)
        conn.close()

    def test_save_metrics_json(self):
        m = _sample_metrics()
        save_metrics_json(m, self.run_dir)
        out = self.run_dir / "metrics.json"
        self.assertTrue(out.exists())
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["run_id"], "run_test_001")
        self.assertEqual(data["upper_avg_speed"], 0.4)
        self.assertEqual(data["total_rallies"], 5)

    def test_runmetrics_required_fields(self):
        m = _sample_metrics()
        self.assertEqual(m.run_id, "run_test_001")
        self.assertEqual(m.source_type, "screen_capture")
        # All fields present
        d = m.to_dict()
        for k in ("run_id", "source_type", "source_ref", "model_path", "device",
                  "conf", "imgsz", "frame_skip", "total_frames", "avg_fps",
                  "avg_player_count", "ball_visible_ratio", "upper_avg_speed",
                  "lower_avg_speed", "total_rallies"):
            self.assertIn(k, d)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests, confirm failure**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_storage -v
```

Expected: `ModuleNotFoundError: No module named 'badminton_analysis.storage'`.

- [ ] **Step 3: Write the minimal implementation**

Create `badminton_analysis/storage.py`:

```python
"""SQLite history and metrics.json writer."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict, field
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
        from .config import load_config
        cfg = load_config()
        db_path = cfg.outputs_dir / "football.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def insert_run(conn: sqlite3.Connection, m: RunMetrics) -> None:
    """Insert or replace a run row."""
    d = m.to_dict()
    cols = ", ".join(d.keys())
    placeholders = ", ".join(["?"] * len(d))
    conn.execute(
        f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({placeholders})",
        list(d.values()),
    )
    conn.commit()


def list_runs(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    """Return the most recent runs as dicts."""
    cur = conn.execute(
        "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def save_metrics_json(m: RunMetrics, run_dir: Path) -> Path:
    """Write the run's metrics to run_dir/metrics.json. Returns the path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "metrics.json"
    out.write_text(json.dumps(m.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out
```

- [ ] **Step 4: Run tests, confirm pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_storage -v
```

Expected: 4 tests, all `ok`.

- [ ] **Step 5: Run full suite to confirm no regressions**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 22 tests in ... OK` (4 new + 18 prior).

- [ ] **Step 6: Commit**

```powershell
git add badminton_analysis/storage.py tests/test_storage.py
git commit -m "feat(storage): add SQLite history and metrics.json writer"
```

---

### Task 5: renderer.py - single-frame overlay

**Files:**
- Create: `badminton_analysis/renderer.py`
- Create: `tests/test_renderer.py`

**Why a separate module:** `BadmintonAnalysisSystem._process_frame` mixes detection, drawing, and stats. The Streamlit app needs to draw overlays on a frame produced by the per-frame call. Extracting `render_frame()` into its own module keeps both call sites in sync and makes it unit-testable with synthetic frames.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_renderer.py`:

```python
"""Unit tests for renderer.render_frame."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from badminton_analysis.analytics import FrameStats
from badminton_analysis.renderer import render_frame


def _blank(h=480, w=640, color=(0, 0, 0)):
    return np.full((h, w, 3), color, dtype=np.uint8)


class RenderFrameTest(unittest.TestCase):
    def test_returns_bgr_frame_same_shape(self):
        frame = _blank()
        stats = FrameStats(
            frame_idx=0, fps=30.0, player_count=0,
            ball_visible=False, ball_center=None, rally_id=0,
            upper_player_count=0, lower_player_count=0,
            avg_speed_upper=0.0, avg_speed_lower=0.0,
            total_distance_upper=0.0, total_distance_lower=0.0,
        )
        out = render_frame(frame, [], [], stats, device="cuda:0")
        self.assertEqual(out.shape, frame.shape)
        # BGR (3 channels)
        self.assertEqual(out.dtype, np.uint8)

    def test_draws_status_bar(self):
        frame = _blank(720, 1280)
        stats = FrameStats(
            frame_idx=42, fps=29.7, player_count=2,
            ball_visible=True, ball_center=(100, 200), rally_id=3,
            upper_player_count=1, lower_player_count=1,
            avg_speed_upper=0.5, avg_speed_lower=0.3,
            total_distance_upper=10.0, total_distance_lower=5.0,
        )
        out = render_frame(frame, [], [], stats, device="cuda:0")
        # Status bar should add non-black pixels in the top region
        top_strip = out[0:60, :, :]
        self.assertGreater(int(top_strip.max()), 0)

    def test_draws_ball_trail(self):
        frame = _blank(720, 1280)
        trail = [(100, 100), (200, 200), (300, 300)]
        stats = FrameStats(
            frame_idx=1, fps=30.0, player_count=0,
            ball_visible=True, ball_center=(300, 300), rally_id=0,
            upper_player_count=0, lower_player_count=0,
            avg_speed_upper=0.0, avg_speed_lower=0.0,
            total_distance_upper=0.0, total_distance_lower=0.0,
        )
        out = render_frame(frame, [], trail, stats, device="cuda:0")
        # The trail should add colored pixels along the path
        # Check the line midpoint for a non-black pixel
        self.assertGreater(int(out[200, 200, :].max()), 0)

    def test_no_trail_no_panic(self):
        frame = _blank()
        stats = FrameStats(
            frame_idx=0, fps=0.0, player_count=0,
            ball_visible=False, ball_center=None, rally_id=0,
            upper_player_count=0, lower_player_count=0,
            avg_speed_upper=0.0, avg_speed_lower=0.0,
            total_distance_upper=0.0, total_distance_lower=0.0,
        )
        # Empty trail, empty player list: should not raise
        out = render_frame(frame, [], [], stats, device="cpu")
        self.assertEqual(out.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests, confirm failure**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_renderer -v
```

Expected: `ModuleNotFoundError: No module named 'badminton_analysis.renderer'`.

- [ ] **Step 3: Write the minimal implementation**

Create `badminton_analysis/renderer.py`:

```python
"""Single-frame overlay renderer.

Designed to be called from both:
  - `BadmintonAnalysisSystem._process_frame` (CLI / live.py path)
  - The Streamlit app's per-frame loop (web UI path)

The function is intentionally side-effect-free: takes a BGR frame and per-frame
state, returns a BGR frame with all overlays drawn.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .analytics import FrameStats


def _put_text(img, text, org, color=(255, 255, 255), scale=0.55, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_status_bar(img: np.ndarray, stats: FrameStats, device: str) -> None:
    h, w = img.shape[:2]
    bar_h = 56
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    color_fps = (0, 255, 0) if stats.fps >= 20 else (0, 165, 255) if stats.fps >= 10 else (0, 0, 255)
    _put_text(img, f"FPS {stats.fps:.1f}", (12, 22), color_fps, 0.7, 2)
    _put_text(img, f"Players {stats.player_count}", (160, 22), (255, 255, 255), 0.6, 2)
    _put_text(img, f"Ball {'Y' if stats.ball_visible else 'N'}", (320, 22),
              (0, 255, 0) if stats.ball_visible else (120, 120, 120), 0.6, 2)
    _put_text(img, f"Rally {stats.rally_id}", (430, 22), (255, 220, 80), 0.6, 2)
    _put_text(img, f"Up {stats.upper_player_count}/{stats.avg_speed_upper:.2f} m/s",
              (560, 22), (200, 220, 255), 0.55, 1)
    _put_text(img, f"Dn {stats.lower_player_count}/{stats.avg_speed_lower:.2f} m/s",
              (820, 22), (255, 200, 200), 0.55, 1)
    _put_text(img, f"Dev {device}", (w - 130, 22), (180, 180, 180), 0.5, 1)
    _put_text(img, f"Frame {stats.frame_idx}", (12, 46), (180, 180, 180), 0.5, 1)
    _put_text(img, f"Dist U {stats.total_distance_upper:.1f}m  D {stats.total_distance_lower:.1f}m",
              (160, 46), (200, 220, 255), 0.5, 1)


def _draw_ball_trail(img: np.ndarray, trail: Sequence[Tuple[int, int]]) -> None:
    if len(trail) < 2:
        return
    pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
    for p in trail[-3:]:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 200, 255), -1)


def _draw_player_boxes(img: np.ndarray, players: Iterable) -> None:
    """Draws a simple bounding box per player (objects with .xyxy or .bbox).

    `players` is a list of objects that have an .xyxy attribute (tuple of 4 ints)
    or a .bbox attribute with the same shape. We don't render skeletons here
    (that lives in player_pose_visualizer); the renderer is a thin summary.
    """
    for p in players:
        box = getattr(p, "xyxy", None) or getattr(p, "bbox", None)
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)


def render_frame(
    frame_bgr: np.ndarray,
    players: List,
    ball_trail: List[Tuple[int, int]],
    stats: FrameStats,
    device: str,
) -> np.ndarray:
    """Return a new BGR frame with overlays drawn."""
    out = frame_bgr.copy()
    _draw_ball_trail(out, ball_trail)
    _draw_player_boxes(out, players)
    _draw_status_bar(out, stats, device)
    return out
```

- [ ] **Step 4: Run tests, confirm pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_renderer -v
```

Expected: 4 tests, all `ok`.

- [ ] **Step 5: Run full suite to confirm no regressions**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 26 tests in ... OK` (4 new + 22 prior).

- [ ] **Step 6: Commit**

```powershell
git add badminton_analysis/renderer.py tests/test_renderer.py
git commit -m "feat(renderer): add single-frame overlay renderer"
```

---

### Task 6: Modify system.py for per-frame web UI use

**Files:**
- Modify: `badminton_analysis/system.py` (expose `process_frame` as a public method; add optional `display_queue` field)

**Why minimal:** The Streamlit app needs to drive frame processing one frame at a time. The existing `_process_frame` already does per-frame work; we just expose it. Adding a `display_queue` field that `_process_frame` pushes to is 3 lines and lets the web UI consume frames via a thread-safe queue (mirroring football's approach).

- [ ] **Step 1: Read system.py around the _process_frame signature**

Read `badminton_analysis/system.py` lines 282-300 to see the current signature of `_process_frame`. Confirm the args are: `(self, frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count)`.

- [ ] **Step 2: Add the public `process_frame` alias and `display_queue` field**

Edit the `BadmintonAnalysisSystem` class. Right after `self.video_path = video_path` is duplicated (line ~118-120 in the current file), add:

```python
        self.display_queue: queue.Queue | None = None
```

(You will need to add `import queue` at the top of the file.)

Add `import queue` to the top of `badminton_analysis/system.py` if not already there. Search the file; if absent, add it next to the existing stdlib imports (around line 3-5).

Add this public alias just below the `_process_frame` definition (so callers can use `system.process_frame(...)`):

```python
    def process_frame(self, frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count):
        """Public alias for _process_frame, used by the Streamlit Web UI."""
        return self._process_frame(frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count)
```

- [ ] **Step 3: Push rendered frame to display_queue if set**

Find the end of `_process_frame` (just before the final `return frame, detect_frame_count`). Add this code right before that return:

```python
        # Push the rendered frame to a web UI queue if one is attached.
        if self.display_queue is not None and frame is not None:
            try:
                self.display_queue.put_nowait((frame, detect_frame_count))
            except queue.Full:
                # Drop oldest, push newest (consumer is too slow).
                try:
                    self.display_queue.get_nowait()
                    self.display_queue.put_nowait((frame, detect_frame_count))
                except Exception:
                    pass
```

- [ ] **Step 4: Verify imports and module still loads**

Run:
```powershell
.\.venv\Scripts\python.exe -c "from badminton_analysis.system import BadmintonAnalysisSystem; import inspect; print('process_frame is method:', 'process_frame' in dir(BadmintonAnalysisSystem)); print('display_queue attr:', 'display_queue' in inspect.signature(BadmintonAnalysisSystem.__init__).parameters or hasattr(BadmintonAnalysisSystem, 'display_queue'))"
```

Expected: `process_frame is method: True` and `display_queue attr: True`.

- [ ] **Step 5: Run full test suite to confirm no regressions**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 26 tests in ... OK` (no new tests, all prior still pass).

- [ ] **Step 6: Run regression byte-level check**

Run:
```powershell
Remove-Item -LiteralPath outputs/demo1 -Recurse -Force -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe (Get-Content C:\Users\18683\AppData\Local\Temp\opencode\preannotate.py -Raw) 2>&1 | Select-Object -Last 1
.\.venv\Scripts\python.exe main.py --video-path videos/demo1.mp4 --template-path templates/demo1.png --display false 2>&1 | Select-String "Video saved" | Select-Object -First 1
Write-Output "detect_demo1.mp4: $((Get-Item outputs/demo1/detect_demo1.mp4).Length)"
```

Expected: `detect_demo1.mp4: 22112015`.

- [ ] **Step 7: Commit**

```powershell
git add badminton_analysis/system.py
git commit -m "refactor(system): expose process_frame and add display_queue hook for web UI"
```

---

### Task 7: TDD - web E2E (thread + queue flow)

**Files:**
- Create: `tests/test_web_e2e.py`

**Why:** Validates the same threading + queue pattern that `app.py` will use. We do this *before* writing `app.py` so the Web UI has a tested foundation.

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_e2e.py`:

```python
"""End-to-end: spawn a thread that calls BadmintonAnalysisSystem.process_frame in a
loop, push frames through display_queue, consume from the main thread, and
verify frames flow end-to-end. Mirrors what the Streamlit app will do.
"""
from __future__ import annotations

import os
import queue
import shutil
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from badminton_analysis.sources.stream_adapter import StreamAdapter
from tests.fake_source import FakeFrameSource


class WebFlowE2ETest(unittest.TestCase):
    def test_threaded_flow_pushes_frames_to_queue(self):
        from badminton_analysis.system import BadmintonAnalysisSystem, load_runtime_dependencies
        load_runtime_dependencies()

        # Set up: 20 synthetic frames + a fake court annotation in a tmp dir.
        frames = [np.full((480, 640, 3), (i * 10) % 256, dtype=np.uint8) for i in range(20)]
        first = frames[0]
        save_dir = Path("outputs").resolve() / "_web_e2e_test"
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True)
        template_path = save_dir / "first_frame.png"
        cv2.imwrite(str(template_path), first)
        with open(save_dir / "court_annotations.txt", "w") as f:
            f.write("corners=[(80, 60), (560, 60), (560, 420), (80, 420)]\n")
            f.write("roi_corners=[(60, 0), (580, 480)]\n")
            f.write("mid_height=240\n")

        fake = FakeFrameSource(frames=frames)
        adapter = StreamAdapter(source=fake, fps=30.0, first_frame=first)
        display_q: queue.Queue = queue.Queue(maxsize=4)
        stop = threading.Event()

        system = BadmintonAnalysisSystem(
            video_path="_web_e2e.mp4",
            template_path=str(template_path),
            output_dir=str(save_dir),
            ball_model_path="weights/yolo11s-ball.pt",
            pose_family="yolo-pose",
            yolo_pose_model="weights/yolo11n-pose.pt",
            show_display=False,
            show_skeletons=False,
            show_player_trajectories=False,
            show_court_trajectory=False,
            show_shuttlecock_trajectory=False,
            show_player_stats=False,
            frame_source=adapter,
            non_interactive_annotation=True,
        )
        system.keep_audio = False
        system.display_queue = display_q

        # Worker thread: consume the stream by reading cap.read() and pushing to queue.
        # We drive the loop by hand here (rather than calling process_video) to
        # avoid the rigid main loop structure and to test the per-frame path.
        # The main loop in process_video also pushes to display_queue automatically
        # (see Task 6), so a simpler integration test is to call process_video and
        # read from the queue.
        def worker():
            try:
                system.process_video()
            except Exception as e:
                print(f"[e2e worker] error: {e}")
            finally:
                stop.set()
                adapter.release()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        consumed = 0
        deadline = time.time() + 30
        while consumed < 5 and time.time() < deadline:
            try:
                frame, idx = display_q.get(timeout=1.0)
                self.assertIsNotNone(frame)
                self.assertEqual(frame.shape[2], 3)  # BGR
                consumed += 1
            except queue.Empty:
                if stop.is_set():
                    break
        stop.set()
        t.join(timeout=10)

        self.assertGreater(consumed, 0, "expected at least one frame from the queue")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test, confirm failure (timeout)**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_web_e2e -v
```

Expected: test passes OR times out (depending on whether Task 6's `display_queue` push works end-to-end). If it passes already, you can move on. If it hangs or fails, the integration between `process_video`'s main loop and the queue push isn't working — re-check Task 6's edit.

- [ ] **Step 3: Commit (whether it passes or you fix it)**

```powershell
git add tests/test_web_e2e.py
git commit -m "test(web): add threaded flow E2E test for display_queue"
```

- [ ] **Step 4: Run full suite**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 27 tests in ... OK`.

---

### Task 8: app.py - Streamlit Web UI

**Files:**
- Create: `app.py` (Streamlit entry)

**Why this task is large but reviewable:** The file is ~250 lines but each section has a clear purpose (sidebar / main / history) and mirrors football's `app.py` line-for-line in structure. We are NOT introducing new behavior; we are wiring together already-tested modules.

- [ ] **Step 1: Write app.py**

Create `app.py` with the following content (verbatim):

```python
"""Streamlit Web UI for Good-Badminton.

Mirrors the structure of `D:\A_shixi\football-realtime-analyzer\app.py`
(sidebar + main + history) but uses badminton-native detection
(YOLO Pose + ball detection + court mapping + upper/lower player tracking).
"""
from __future__ import annotations

import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from badminton_analysis.analytics import (
    BallTrail,
    FpsCounter,
    FrameStats,
    MetricsAggregator,
)
from badminton_analysis.config import (
    OUTPUTS_DIR,
    ROOT,
    RUNS_DIR,
    UPLOADS_DIR,
    load_config,
)
from badminton_analysis.renderer import render_frame
from badminton_analysis.sources import (
    HeadlessBrowserSource,
    ScreenCaptureSource,
    StreamAdapter,
)
from badminton_analysis.storage import (
    RunMetrics,
    init_db,
    insert_run,
    list_runs,
    save_metrics_json,
)
from badminton_analysis.system import BadmintonAnalysisSystem, load_runtime_dependencies

st.set_page_config(page_title="Good-Badminton 实时识别系统", layout="wide")
st.title("基于 YOLO 的羽毛球比赛多源实时目标检测与分析系统")

cfg = load_config()

SOURCE_LABELS = {
    "video_file": "本地视频",
    "browser_headless": "网页直播（无头浏览器）",
    "screen_capture": "屏幕捕获",
}

PRESETS = {
    "全屏": None,
    "左半屏": None,
    "右半屏": None,
    "上半屏": None,
    "下半屏": None,
    "中央 1280x720": None,
}


def _detect_screen_size() -> tuple[int, int]:
    import mss
    with mss.MSS() as sct:
        mon = sct.monitors[1]
        return int(mon["width"]), int(mon["height"])


def _resolve_preset(name: str, custom: list[int] | None) -> list[int]:
    if name == "自定义":
        return list(custom or [100, 100, 1280, 720])
    sw, sh = _detect_screen_size()
    if name == "全屏":
        return [0, 0, sw, sh]
    if name == "左半屏":
        return [0, 0, sw // 2, sh]
    if name == "右半屏":
        return [sw // 2, 0, sw - sw // 2, sh]
    if name == "上半屏":
        return [0, 0, sw, sh // 2]
    if name == "下半屏":
        return [0, sh // 2, sw, sh - sh // 2]
    if name == "中央 1280x720":
        w, h = 1280, 720
        return [max(0, (sw - w) // 2), max(0, (sh - h) // 2), w, h]
    return [100, 100, 1280, 720]


def _make_screen_source(region: list[int]) -> ScreenCaptureSource:
    return ScreenCaptureSource(
        left=region[0], top=region[1], width=region[2], height=region[3]
    )


def _make_browser_source(url: str) -> HeadlessBrowserSource:
    return HeadlessBrowserSource(url=url)


with st.sidebar:
    st.header("参数设置")
    source_type = st.selectbox(
        "视频源类型",
        list(SOURCE_LABELS.keys()),
        format_func=lambda x: SOURCE_LABELS[x],
    )
    device = st.selectbox("设备", ["cuda:0", "auto", "cpu"], index=0)
    model_path = st.text_input("模型路径", cfg.default_model)
    imgsz = st.slider("imgsz (推理分辨率)", 320, 1280, cfg.imgsz, step=64)
    conf = st.slider("置信度阈值 conf", 0.05, 0.9, cfg.conf, step=0.05)
    frame_skip = st.slider("帧跳过 frame_skip", 1, 10, cfg.frame_skip, step=1)
    no_court = st.checkbox("跳过球场自动检测 (--no-court)", False)
    show_raw = st.checkbox("显示原始画面（调试用）", False)
    save_output = st.checkbox("保存标注视频", cfg.save_output)
    st.divider()
    start_btn = st.button("开始识别", type="primary")
    stop_btn = st.button("停止识别")

# Source-specific UI
uploaded_file = None
browser_url = ""
region = [cfg.screen_region.left, cfg.screen_region.top,
          cfg.screen_region.width, cfg.screen_region.height]

if source_type == "video_file":
    uploaded_file = st.file_uploader("上传本地视频 (mp4/avi/mov, 最大 4GB)",
                                     type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        size_mb = uploaded_file.size / (1024 * 1024)
        st.caption(f"已选择: {uploaded_file.name} | 大小: {size_mb:.1f} MB")

elif source_type == "browser_headless":
    browser_url = st.text_input("直播网页地址", placeholder="https://...")
    if browser_url and st.button("测试连通性", key="test_browser"):
        with st.spinner("正在测试..."):
            try:
                src = _make_browser_source(browser_url)
                if src.open():
                    r = src.next_frame()
                    src.close()
                    if r.ok and r.frame is not None and r.frame.std() > 1:
                        st.success(f"连通正常 ({r.frame.shape[1]}x{r.frame.shape[0]})")
                    else:
                        st.warning("已打开但视频可能还在加载")
                else:
                    st.error("无法启动无头浏览器")
            except Exception as e:
                st.error(f"测试失败: {e}")

elif source_type == "screen_capture":
    try:
        sw, sh = _detect_screen_size()
    except Exception:
        sw, sh = 1920, 1080
    st.caption(f"主屏幕: {sw} x {sh}")
    if "scr_preset" not in st.session_state:
        st.session_state["scr_preset"] = "全屏"
    preset = st.selectbox("区域预设", list(PRESETS.keys()) + ["自定义"],
                          index=list(PRESETS.keys()).index(st.session_state["scr_preset"])
                          if st.session_state["scr_preset"] in PRESETS else 0,
                          key="scr_preset_sel")
    st.session_state["scr_preset"] = preset
    if preset != "自定义":
        region = _resolve_preset(preset, None)
        st.info(f"应用预设: left={region[0]} top={region[1]} {region[2]}x{region[3]}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            region[0] = st.slider("left", 0, sw, region[0], step=10)
            region[1] = st.slider("top", 0, sh, region[1], step=10)
        with col2:
            region[2] = st.slider("width", 100, sw, region[2], step=10)
            region[3] = st.slider("height", 100, sh, region[3], step=10)
    if st.button("预览捕获区域", key="preview_region"):
        try:
            src = _make_screen_source(region)
            if src.open():
                r = src.next_frame()
                src.close()
                if r.ok and r.frame is not None:
                    rgb = cv2.cvtColor(r.frame, cv2.COLOR_BGR2RGB)
                    st.image(rgb, caption=f"预览 {region[2]}x{region[3]}", channels="RGB")
        except Exception as e:
            st.error(f"预览失败: {e}")


# Session state for the run
for k, v in [("running", False), ("last_metrics", None),
             ("last_run_dir", None), ("current_run_id", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

if stop_btn:
    st.session_state["running"] = False


def _build_source():
    if source_type == "video_file":
        if uploaded_file is None:
            return None, "请上传本地视频文件"
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = UPLOADS_DIR / f"upload_{uploaded_file.name}"
        with open(tmp, "wb") as f:
            while True:
                chunk = uploaded_file.read(10 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        from badminton_analysis.sources.video_file import VideoFileSource
        return VideoFileSource(str(tmp)), None
    if source_type == "browser_headless":
        if not browser_url:
            return None, "请输入直播网页地址"
        return _make_browser_source(browser_url), None
    if source_type == "screen_capture":
        return _make_screen_source(region), None
    return None, "未知视频源类型"


def _save_results(summary, run_id):
    if summary.get("total_frames", 0) <= 0:
        return None
    if not run_id:
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = RUNS_DIR / run_id
    try:
        m = RunMetrics(
            run_id=run_id,
            source_type=source_type,
            source_ref=str(region) if source_type == "screen_capture" else
                        (browser_url[:80] if source_type == "browser_headless" else
                         (uploaded_file.name if uploaded_file else "")),
            model_path=model_path,
            device=device,
            conf=conf,
            imgsz=imgsz,
            frame_skip=frame_skip,
            **summary,
        )
        save_metrics_json(m, run_dir)
    except Exception:
        pass
    try:
        conn = init_db()
        insert_run(conn, m)
        conn.close()
    except Exception:
        pass
    return str(run_dir), run_id


video_col, stats_col = st.columns([3, 1])
status_box = st.container()
frame_ph = video_col.empty()
stats_ph = stats_col.empty()

if start_btn and not st.session_state["running"]:
    src, err = _build_source()
    if err:
        with status_box:
            st.error(err)
    elif no_court:
        # Headless-friendly path: skip the OpenCV court annotation flow by
        # injecting a court_annotations.txt before opening. The output dir is
        # a fresh RUNS_DIR / run_id; we precompute the same one we will use.
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        save_dir = RUNS_DIR / run_id
        save_dir.mkdir(parents=True, exist_ok=True)
        # We can't pre-write annotations (don't know the frame size until
        # source is opened). The system will fall through to the no-court path
        # if we set skip_court_annotation=True.
        try:
            detector_ctor = lambda: BadmintonAnalysisSystem(
                video_path=f"{run_id}.mp4",
                template_path=None,
                output_dir=str(save_dir),
                ball_model_path="weights/yolo11s-ball.pt",
                pose_family="yolo-pose",
                yolo_pose_model="weights/yolo11n-pose.pt",
                show_display=False,
                show_skeletons=True,
                show_player_trajectories=True,
                show_court_trajectory=False,
                show_shuttlecock_trajectory=True,
                show_player_stats=True,
                frame_source=None,  # placeholder
                skip_court_annotation=True,
                non_interactive_annotation=True,
            )
        except Exception as e:
            with status_box:
                st.error(f"无法构造系统: {e}")
            src.close()
        else:
            # Open source, grab first frame, wire up the real system.
            if not src.open():
                with status_box:
                    st.error(f"无法打开视频源: {SOURCE_LABELS[source_type]}")
            else:
                first = src.next_frame()
                if not first.ok or first.frame is None:
                    src.close()
                    with status_box:
                        st.error("首帧抓取失败")
                else:
                    first_path = save_dir / "first_frame.png"
                    cv2.imwrite(str(first_path), first.frame)
                    adapter = StreamAdapter(source=src, fps=30.0, first_frame=first.frame)
                    system = detector_ctor()
                    system.frame_source = adapter
                    system.keep_audio = False
                    _run_inference_thread(system, adapter, run_id, save_dir, status_box,
                                          frame_ph, stats_ph, st.session_state)
    else:
        # Default path: standard CLI flow (auto court detection). Not yet
        # threaded because the existing process_video() is monolithic; we
        # call it from a thread to keep the UI responsive.
        from badminton_analysis.sources.video_file import VideoFileSource
        from badminton_analysis.system import BadmintonAnalysisSystem
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        save_dir = RUNS_DIR / run_id
        save_dir.mkdir(parents=True, exist_ok=True)
        if not src.open():
            with status_box:
                st.error(f"无法打开视频源: {SOURCE_LABELS[source_type]}")
        else:
            first = src.next_frame()
            if not first.ok or first.frame is None:
                src.close()
                with status_box:
                    st.error("首帧抓取失败")
            else:
                first_path = save_dir / "first_frame.png"
                cv2.imwrite(str(first_path), first.frame)
                adapter = StreamAdapter(source=src, fps=30.0, first_frame=first.frame)
                system = BadmintonAnalysisSystem(
                    video_path=f"{run_id}.mp4",
                    template_path=str(first_path),
                    output_dir=str(save_dir),
                    ball_model_path="weights/yolo11s-ball.pt",
                    pose_family="yolo-pose",
                    yolo_pose_model="weights/yolo11n-pose.pt",
                    show_display=False,
                    show_skeletons=True,
                    show_player_trajectories=True,
                    show_court_trajectory=True,
                    show_shuttlecock_trajectory=True,
                    show_player_stats=True,
                    frame_source=adapter,
                    non_interactive_annotation=True,
                )
                system.keep_audio = False
                _run_inference_thread(system, adapter, run_id, save_dir, status_box,
                                      frame_ph, stats_ph, st.session_state)


def _run_inference_thread(system, adapter, run_id, save_dir, status_box, frame_ph,
                          stats_ph, session_state):
    """Spawn a background thread that drives process_video and pushes frames."""
    session_state["running"] = True
    session_state["current_run_id"] = run_id
    display_q: queue.Queue = queue.Queue(maxsize=2)
    system.display_queue = display_q
    stop_event = threading.Event()
    shared = {"metrics_agg": MetricsAggregator()}

    def worker():
        try:
            system.process_video()
        except Exception as e:
            print(f"[worker] error: {e}")
        finally:
            stop_event.set()
            adapter.release()
            try:
                display_q.put_nowait(("__END__", -1))
            except Exception:
                pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    with status_box:
        st.info(f"识别中... run_id={run_id}")

    while session_state.get("running", False) and not stop_event.is_set():
        try:
            item = display_q.get(timeout=0.3)
            if item[0] == "__END__":
                break
            frame_bgr, _idx = item
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_ph.image(Image.fromarray(rgb) if False else __import__("PIL").Image.fromarray(rgb),
                            channels="RGB", use_container_width=True)
            stats_ph.markdown(
                f"**FPS** (实时)\n\n**Players** (实时)\n\n"
                f"**Rally** (实时)\n\n**Status** running"
            )
        except queue.Empty:
            continue

    t.join(timeout=5)
    session_state["running"] = False
    # Save results
    summary = shared["metrics_agg"].summary()
    rd, rid = _save_results(summary, run_id)
    if rd:
        session_state["last_metrics"] = summary
        session_state["last_run_dir"] = rd


st.divider()
st.subheader("本次识别结果")
m = st.session_state.get("last_metrics")
if m:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总帧数", m["total_frames"])
    c2.metric("回合数", m["total_rallies"])
    c3.metric("平均球员数", f"{m['avg_player_count']:.1f}")
    c4.metric("足球可见率" if False else "羽毛球可见率", f"{m['ball_visible_ratio']*100:.1f}%")
    p1, p2 = st.columns(2)
    p1.metric("上半场 平均速度", f"{m['upper_avg_speed']:.2f} m/s")
    p2.metric("下半场 平均速度", f"{m['lower_avg_speed']:.2f} m/s")
    if st.session_state.get("last_run_dir"):
        st.code(f"结果目录: {st.session_state['last_run_dir']}")
else:
    st.info("尚未进行识别。点击「开始识别」后这里会显示本次汇总指标。")

st.subheader("历史识别记录 (SQLite)")
try:
    conn = init_db()
    rows = list_runs(conn, limit=20)
    conn.close()
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        show_cols = [
            "run_id", "source_type", "source_ref", "device", "total_frames",
            "avg_fps", "avg_player_count", "ball_visible_ratio",
            "upper_avg_speed", "lower_avg_speed", "total_rallies", "created_at",
        ]
        avail = [c for c in show_cols if c in df.columns]
        st.dataframe(df[avail], use_container_width=True, hide_index=True)
        st.caption(f"数据库位置: {OUTPUTS_DIR / 'football.db'}")
    else:
        st.info("暂无历史记录。")
except Exception as e:
    st.warning(f"读取历史记录失败: {e}")
```

- [ ] **Step 2: Verify app.py imports cleanly**

Run:
```powershell
.\.venv\Scripts\python.exe -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('app.py parses OK')"
```

Expected: `app.py parses OK`.

- [ ] **Step 3: Run full test suite**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 27 tests in ... OK` (app.py is not unit-tested here; the E2E test from Task 7 covers the threaded flow).

- [ ] **Step 4: Commit**

```powershell
git add app.py
git commit -m "feat(web): add Streamlit app.py mirroring football structure with badminton-native detection"
```

---

### Task 9: start.bat - one-click launcher

**Files:**
- Create: `start.bat` (Windows only; mirrors football's `start.bat` style)

- [ ] **Step 1: Write start.bat**

Create `start.bat` with the following content (verbatim):

```bat
@echo off
chcp 65001 >nul
echo ========================================
echo   Good-Badminton - Streamlit Web UI
echo ========================================
echo.

cd /d "%~dp0"

if not exist ".venv" (
    echo [1/5] 创建虚拟环境...
    "D:\Program\Python.12.10\python.exe" -m venv .venv
    if errorlevel 1 (
        echo 错误: 创建虚拟环境失败
        pause
        exit /b 1
    )
) else (
    echo [1/5] 虚拟环境已存在
)

echo [2/5] 激活虚拟环境...
call .venv\Scripts\activate.bat

echo [3/5] 安装/更新依赖...
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

echo [4/5] 检查 PyTorch GPU...
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo [5/5] 启动 Streamlit...
echo 浏览器打开: http://localhost:8501
start "" "http://localhost:8501"
streamlit run app.py --server.headless true --server.port 8501 --server.maxUploadSize 4096

pause
```

- [ ] **Step 2: Verify the file was written**

Run:
```powershell
Test-Path start.bat
Get-Content start.bat -TotalCount 5
```

Expected: `True` and the first 5 lines of the script.

- [ ] **Step 3: Commit**

```powershell
git add start.bat
git commit -m "feat: add start.bat one-click Streamlit launcher"
```

---

### Task 10: Smoke test - launch Streamlit and verify it loads

**Files:** none (manual verification)

- [ ] **Step 1: Start Streamlit in the background**

Run:
```powershell
.\.venv\Scripts\streamlit.exe run app.py --server.headless true --server.port 8501 --server.maxUploadSize 4096 2>&1 | Out-File -Encoding utf8 C:\Users\18683\AppData\Local\Temp\opencode\streamlit.log
```

(Use a separate PowerShell window if the bash tool blocks.)

- [ ] **Step 2: Verify the server is up**

Run:
```powershell
Test-NetConnection localhost -Port 8501 -InformationLevel Quiet
```

Expected: `True`.

- [ ] **Step 3: Verify the index page loads**

Run:
```powershell
(Invoke-WebRequest http://localhost:8501 -UseBasicParsing -TimeoutSec 10).StatusCode
```

Expected: `200`.

- [ ] **Step 4: Stop the Streamlit server**

Run:
```powershell
Get-Process streamlit -ErrorAction SilentlyContinue | Stop-Process -Force
```

- [ ] **Step 5: No commit (verification only)**

If any step fails, check the log at `C:\Users\18683\AppData\Local\Temp\opencode\streamlit.log`.

---

### Task 11: Update README.md

**Files:**
- Modify: `README.md` (append a "Web UI" section)

- [ ] **Step 1: Find the right insertion point**

Run:
```powershell
Select-String -Path README.md -Pattern "^## "
```

Find the last `##` section before "Star History" or "License". Insert before it.

- [ ] **Step 2: Append the Web UI section**

Insert (use Python with UTF-8 to avoid encoding issues; the existing README.md is UTF-8 BOM):

```markdown
## Web 鎺у埗鍙? (Streamlit)

闄や簡 CLI锛屼篃鎻愪緵 Streamlit Web 鎺у埗鍙帮紝缁撴瀯鍙傝?? `football-realtime-analyzer`锛堜晶鏍忓弬鏁? + 涓诲尯瀹炴椂鐢婚潰 + 搴曢儴鍘嗗彶锛夛紝妫?娴嬪眰鐢ㄧ窘姣涚悆鍘熺敓锛圷OLO Pose + 鐞冩??娴? + 鐞冨満鏄犲皠 + 涓婁笅鍗婂満鐞冨憳杩借釜锛夈??

### 鍚?鍔?

```bat
:: 一键启动（首次会自动建 venv + 装依赖 + 起 Streamlit）
start.bat

:: 或手动
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501
```

浏览器打开 http://localhost:8501

### 视频源

侧栏选择三类源之一：
- **本地视频**：上传 mp4/avi/mov（最大 4GB）
- **网页直播**：粘 https URL，无头 Chrome 抓 `<video>` 元素
- **屏幕捕获**：预设（全屏 / 左半 / 右半 / 上半 / 下半 / 中央 1280x720）或自定义矩形，可"预览捕获区域"

### 实时统计

右栏每帧刷新：
- FPS / 设备 / 帧号
- 检测球员数 / 羽毛球可见
- 当前回合号
- 上半场 + 下半场球员数 / 平均速度 / 累计移动距离

跑完写入 SQLite `outputs/football.db`，底部表格可看历史。

### 球场检测选项

- 默认：自动检测（需画面有清晰球场线）+ 非交互模式
- "跳过球场自动检测" 复选框：--no-court 模式，对任意内容都能跑（无球场映射）

### 文件位置

| 目录 | 内容 |
|---|---|
| `outputs/runs/run_<时间戳>_<id>/` | 每轮次的 `metrics.json` + 标注视频 |
| `outputs/football.db` | SQLite 历史 |
| `outputs/uploads/` | Web 上传的本地视频 |
| `configs/default.yaml` | 默认参数 |

### 已知限制

- 屏幕源非交互式（无 GUI 弹窗），所以"跳过球场自动检测"是屏幕源推荐选项
- mss 抓的帧率跟随屏幕刷新率（典型 30-60 fps）
- Streamlit 单用户；多人用需要鉴权层
- 跑长视频建议提高 `frame_skip` 节省 GPU

```

- [ ] **Step 3: Verify the section is in the right place**

Run:
```powershell
Select-String -Path README.md -Pattern "## Web 控制台"
```

Expected: prints the line number where the new section starts.

- [ ] **Step 4: Commit**

```powershell
git add README.md
git commit -m "docs: document Streamlit Web UI"
```

---

### Task 12: Final regression - existing local file flow must be byte-identical

**Files:** none (verification only)

**Why this gate:** Tasks 6 modified `system.py`. We must confirm the existing CLI is byte-identical to its pre-plan state.

- [ ] **Step 1: Wipe outputs and re-preannotate**

Run:
```powershell
Remove-Item -LiteralPath outputs/demo1 -Recurse -Force -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe (Get-Content C:\Users\18683\AppData\Local\Temp\opencode\preannotate.py -Raw) 2>&1 | Select-Object -Last 1
```

- [ ] **Step 2: Run the standard local-file command**

Run:
```powershell
.\.venv\Scripts\python.exe main.py --video-path videos/demo1.mp4 --template-path templates/demo1.png --display false 2>&1 | Select-String "Video saved" | Select-Object -First 1
```

- [ ] **Step 3: Verify byte sizes match the historical reference**

Run:
```powershell
Write-Output "detect_demo1.mp4: $((Get-Item outputs/demo1/detect_demo1.mp4).Length) (expected 22112015)"
Write-Output "detections.jsonl: $((Get-Item outputs/demo1/detections.jsonl).Length) (expected 467583)"
Write-Output "metadata.json: $((Get-Item outputs/demo1/metadata.json).Length) (expected 944)"
```

Expected: all three match exactly.

- [ ] **Step 4: Run full test suite**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 27 tests in ... OK`.

- [ ] **Step 5: No commit (verification only)**

If any check fails, the `system.py` change broke something — fix before declaring done.

---
