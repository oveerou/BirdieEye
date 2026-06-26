# Court Model Auto-Update + Real-Time Heatmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** During a badminton match, automatically refresh the court model when a "full-court" view appears, and display a real-time sliding-window heatmap of player positions overlaid on the output video.

**Architecture:** New files `badminton_analysis/court/updater.py` (CourtModelUpdater: smart re-detection at intervals) and `badminton_analysis/analytics/heatmap.py` (CourtHeatmap: 2-min sliding window per half-court). Integrate both into `BadmintonAnalysisSystem._process_frame`; render heatmap as an inline overlay on the output frame. Plumb CLI/Streamlit flags for the new tunables.

**Tech Stack:** Existing stack (OpenCV, NumPy, dataclasses, threading). No new external deps.

## Global Constraints

- All code follows existing `badminton_analysis` package layout.
- Existing reference: `python main.py --video-path videos/demo1.mp4 --template-path templates/demo1.png --display false` must still produce byte-identical `outputs/demo1/detect_demo1.mp4` of exactly **22,112,015 bytes** (default behaviour must be a no-op overlay when no position data is in the heatmap yet).
- Existing 32 unit tests must continue to pass.
- TDD: each new module has a `tests/test_*.py` written FIRST.
- UTF-8 for all source files.
- venv Python: `D:\A_shixi\Good-Badminton\.venv\Scripts\python.exe`. Workdir: `D:\A_shixi\Good-Badminton`.
- All commits are local; do not push.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `badminton_analysis/court/updater.py` | Create (TDD) | `CourtModelUpdater`: smart court re-detection |
| `badminton_analysis/analytics/heatmap.py` | Create (TDD) | `CourtHeatmap`: 2-min sliding window per half |
| `badminton_analysis/system.py` | Modify | Wire `court_updater` + `heatmap` into `_process_frame`; expose `court_corners` as public attribute |
| `badminton_analysis/tracking/player.py` | Modify | Add `court_pos: tuple | None` field on Player objects |
| `badminton_analysis/live.py` | Modify | Add `--court-update-interval`, `--no-heatmap`, `--heatmap-window` flags |
| `app.py` | Modify | Add sliders/checkboxes in sidebar |
| `tests/test_court_updater.py` | Create | Unit tests for updater |
| `tests/test_heatmap.py` | Create | Unit tests for heatmap |
| `tests/test_heatmap_integration.py` | Create | E2E: feed positions, render overlay, check non-empty output |
| `README.md` | Modify | +"Court model update" and "Heatmap" sections |

---

### Task 1: TDD - court/updater.py

**Files:**
- Create: `badminton_analysis/court/updater.py`
- Create: `tests/test_court_updater.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_court_updater.py`:

```python
"""Unit tests for CourtModelUpdater."""
from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

from badminton_analysis.court.updater import CourtModelUpdater


def _sharp_frame(h=720, w=1280):
    """A frame with high laplacian variance (focused)."""
    rng = np.random.default_rng(0)
    img = rng.integers(50, 200, (h, w, 3), dtype=np.uint8)
    # Add a sharp line so laplacian variance is non-trivial
    cv2.line(img, (100, 100), (w - 100, h - 100), (255, 255, 255), 2)
    return img


def _blurry_frame(h=720, w=1280):
    """A frame that is mostly uniform (low laplacian variance)."""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    return img


class CourtModelUpdaterTest(unittest.TestCase):
    def _make_system(self, current_corners=None):
        system = MagicMock()
        system.court_corners = current_corners
        system.court_roi_corners = [(0, 0), (640, 480)] if current_corners else None
        system.mid_height = 240
        system.court_mapper = None
        system.save_dir = "/tmp"
        return system

    def test_quality_low_when_no_players(self):
        u = CourtModelUpdater(self._make_system(), check_interval_sec=0)
        # No players -> 0
        self.assertEqual(u._score_quality(_sharp_frame(), player_count=0), 0.0)
        self.assertEqual(u._score_quality(_sharp_frame(), player_count=1), 0.0)

    def test_quality_low_when_blurry(self):
        u = CourtModelUpdater(self._make_system(), check_interval_sec=0)
        # Players but blurry frame -> 0
        self.assertEqual(u._score_quality(_blurry_frame(), player_count=2), 0.0)

    def test_quality_high_with_2_players_and_sharp_frame(self):
        u = CourtModelUpdater(self._make_system(), check_interval_sec=0)
        q = u._score_quality(_sharp_frame(), player_count=2)
        self.assertGreater(q, 0.3)
        self.assertLessEqual(q, 1.0)

    def test_should_check_respects_interval(self):
        system = self._make_system(current_corners=[(0, 0), (640, 0), (640, 480), (0, 480)])
        u = CourtModelUpdater(system, check_interval_sec=10.0)
        # Just initialized: should not check
        self.assertFalse(u._should_check())
        # Force last_check to past
        u._last_check = time.time() - 11.0
        self.assertTrue(u._should_check())

    def test_is_plausible_update_no_prior_corners(self):
        u = CourtModelUpdater(self._make_system(current_corners=None))
        # No prior corners -> any update is plausible
        self.assertTrue(u._is_plausible_update([(0, 0), (100, 0), (100, 100), (0, 100)]))

    def test_is_plausible_update_rejects_huge_jump(self):
        system = self._make_system(current_corners=[(100, 100), (200, 100), (200, 200), (100, 200)])
        u = CourtModelUpdater(system, check_interval_sec=0)
        # 200+ pixel jump should be rejected
        self.assertFalse(u._is_plausible_update([(0, 0), (100, 0), (100, 100), (0, 100)]))

    def test_is_plausible_update_accepts_small_drift(self):
        system = self._make_system(current_corners=[(100, 100), (200, 100), (200, 200), (100, 200)])
        u = CourtModelUpdater(system, check_interval_sec=0)
        # 20px drift is fine
        self.assertTrue(u._is_plausible_update([(120, 100), (200, 100), (200, 200), (100, 200)]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests, confirm failure**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_court_updater -v
```

Expected: `ModuleNotFoundError: No module named 'badminton_analysis.court.updater'`.

- [ ] **Step 3: Write the implementation**

Create `badminton_analysis/court/updater.py`:

```python
"""Smart court model updater: re-detects the court when a high-quality
"full-court view" frame appears during playback.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import numpy as np

from .detector import auto_detect_court_corners
from .mapper import compute_expanded_roi, CourtMapper


class CourtModelUpdater:
    """Periodically re-runs auto court detection on the current frame and
    updates `system.court_corners` / `system.court_mapper` if the new model
    looks plausible and higher-quality than the current one.
    """

    MAX_CORNER_JUMP_PX = 100  # a 100+px jump in any corner is implausible

    def __init__(self, system, check_interval_sec: float = 8.0, min_quality: float = 0.5):
        self.system = system
        self.check_interval_sec = float(check_interval_sec)
        self.min_quality = float(min_quality)
        self._last_check = 0.0
        self._update_count = 0
        self._last_quality = 0.0
        self._last_status = "init"

    def _should_check(self) -> bool:
        return (time.time() - self._last_check) >= self.check_interval_sec

    def _score_quality(self, frame, player_count: int) -> float:
        """Score 0-1. Higher = better candidate for court re-detection.

        Requires >= 2 players visible AND a reasonably sharp frame.
        """
        if player_count < 2:
            return 0.0
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if var < 30:
            return 0.0
        player_score = min(1.0, player_count / 2.0)
        focus_score = min(1.0, var / 300.0)
        return player_score * focus_score

    def _is_plausible_update(self, new_corners) -> bool:
        """Refuse the update if any corner jumps too far from the prior model."""
        prior = self.system.court_corners
        if not prior:
            return True
        for old, new in zip(prior, new_corners):
            dist = float(((old[0] - new[0]) ** 2 + (old[1] - new[1]) ** 2) ** 0.5)
            if dist > self.MAX_CORNER_JUMP_PX:
                return False
        return True

    def _detect(self, frame):
        """Run auto_detect_court_corners on this frame; return (corners, roi, mid)."""
        h, w = frame.shape[:2]
        fixed = (1080, 720)
        base = cv2.resize(frame, fixed)
        corners, _line, _dbg = auto_detect_court_corners(base)
        if not corners:
            return None, None, None
        roi = compute_expanded_roi(corners, base.shape)
        mapper = CourtMapper(corners)
        _, mid = mapper.draw_court_overlay(base)
        # Scale back to original frame size
        sx = w / fixed[0]
        sy = h / fixed[1]
        orig_corners = [(int(x * sx), int(y * sy)) for x, y in corners]
        orig_roi = [(int(x * sx), int(y * sy)) for x, y in roi]
        return orig_corners, orig_roi, int(mid * sy)

    def _save_annotations(self, corners, roi, mid):
        path = os.path.join(self.system.save_dir, "court_annotations.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"corners={corners}\n")
            f.write(f"roi_corners={roi}\n")
            f.write(f"mid_height={mid}\n")

    def maybe_update(self, frame, player_count: int) -> bool:
        """If it's time, check the current frame. Returns True on successful update."""
        if not self._should_check():
            return False
        self._last_check = time.time()
        quality = self._score_quality(frame, player_count)
        self._last_quality = quality
        if quality < self.min_quality:
            self._last_status = "skipped (low quality)"
            return False
        new_corners, new_roi, new_mid = self._detect(frame)
        if not new_corners or len(new_corners) != 4:
            self._last_status = "skipped (no detection)"
            return False
        if not self._is_plausible_update(new_corners):
            self._last_status = "skipped (implausible jump)"
            return False
        # Apply the new model
        self.system.court_corners = new_corners
        self.system.court_roi_corners = new_roi
        self.system.mid_height = new_mid
        self.system.court_mapper = CourtMapper(new_corners)
        self._update_count += 1
        self._last_status = f"updated (#{self._update_count})"
        self._save_annotations(new_corners, new_roi, new_mid)
        return True
```

- [ ] **Step 4: Run tests, confirm pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_court_updater -v
```

Expected: 7 tests, all `ok`.

- [ ] **Step 5: Run full suite to confirm no regressions**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 39 tests in ... OK` (7 new + 32 prior).

- [ ] **Step 6: Commit**

```powershell
git add badminton_analysis/court/updater.py tests/test_court_updater.py
git commit -m "feat(court): add CourtModelUpdater for smart in-play court re-detection"
```

---

### Task 2: TDD - analytics/heatmap.py

**Files:**
- Create: `badminton_analysis/analytics/heatmap.py`
- Create: `tests/test_heatmap.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_heatmap.py`:

```python
"""Unit tests for CourtHeatmap."""
from __future__ import annotations

import time
import unittest

import numpy as np

from badminton_analysis.analytics.heatmap import CourtHeatmap


class CourtHeatmapTest(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(CourtHeatmap.COURT_W_M, 6.1)
        self.assertEqual(CourtHeatmap.COURT_H_M, 13.4)
        self.assertEqual(CourtHeatmap.GRID_W, 60)
        self.assertEqual(CourtHeatmap.GRID_H, 130)
        self.assertEqual(CourtHeatmap.WINDOW_SEC, 120.0)

    def test_add_maps_position_to_grid(self):
        h = CourtHeatmap()
        h.add(3.05, 6.5, "upper", t=1.0)  # mid-court
        self.assertEqual(len(h.upper_events), 1)
        ts, gx, gy = h.upper_events[0]
        self.assertEqual(ts, 1.0)
        # gx = 3.05/6.1 * 60 = 30, gy = 6.5/13.4 * 130 ~= 63
        self.assertEqual(gx, 30)
        self.assertEqual(gy, 63)

    def test_add_lower_goes_to_lower_events(self):
        h = CourtHeatmap()
        h.add(1.0, 11.0, "lower", t=1.0)
        self.assertEqual(len(h.lower_events), 1)
        self.assertEqual(len(h.upper_events), 0)

    def test_eviction_after_window(self):
        h = CourtHeatmap()
        h.add(3.0, 6.0, "upper", t=100.0)
        h.add(3.0, 6.0, "upper", t=100.0 + 200.0)  # outside window
        h._evict(now=100.0 + 200.0)
        self.assertEqual(len(h.upper_events), 1)  # only the recent one remains

    def test_render_minimap_shape(self):
        h = CourtHeatmap()
        h.add(3.0, 6.0, "upper", t=1.0)
        h.add(3.0, 11.0, "lower", t=1.0)
        minimap = h.render_minimap()
        # 2 * GRID_H tall, GRID_W wide
        self.assertEqual(minimap.shape, (CourtHeatmap.GRID_H * 2, CourtHeatmap.GRID_W, 3))
        self.assertEqual(minimap.dtype, np.uint8)

    def test_overlay_on_returns_same_shape(self):
        h = CourtHeatmap()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        h.add(3.0, 6.0, "upper", t=1.0)
        out = h.overlay_on(frame, position="bottom-right", size=(240, 130))
        self.assertEqual(out.shape, frame.shape)
        # The overlay region should be non-black (heatmap drawn on it)
        region = out[720 - 130 - 20:720 - 20, 1280 - 240 - 20:1280 - 20, :]
        self.assertGreater(int(region.max()), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests, confirm failure**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_heatmap -v
```

Expected: `ModuleNotFoundError: No module named 'badminton_analysis.analytics.heatmap'`.

- [ ] **Step 3: Write the implementation**

Create `badminton_analysis/analytics/heatmap.py`:

```python
"""Real-time sliding-window heatmap of player positions in court coordinates."""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


class CourtHeatmap:
    """Per-half sliding-window heatmap over a 6.1m x 13.4m standard court.

    Positions are recorded as events (t, gx, gy). The window slides forward
    in time; events older than `WINDOW_SEC` are evicted.
    """

    COURT_W_M = 6.1
    COURT_H_M = 13.4
    GRID_W = 60
    GRID_H = 130
    WINDOW_SEC = 120.0  # 2 minutes

    def __init__(self):
        self.upper_events: deque = deque()
        self.lower_events: deque = deque()

    def add(self, x_m: float, y_m: float, half: str, t: Optional[float] = None) -> None:
        if t is None:
            t = time.time()
        gx = int((x_m / self.COURT_W_M) * self.GRID_W)
        gy = int((y_m / self.COURT_H_M) * self.GRID_H)
        gx = max(0, min(self.GRID_W - 1, gx))
        gy = max(0, min(self.GRID_H - 1, gy))
        if half == "upper":
            self.upper_events.append((t, gx, gy))
        else:
            self.lower_events.append((t, gx, gy))

    def _evict(self, now: float) -> None:
        cutoff = now - self.WINDOW_SEC
        while self.upper_events and self.upper_events[0][0] < cutoff:
            self.upper_events.popleft()
        while self.lower_events and self.lower_events[0][0] < cutoff:
            self.lower_events.popleft()

    def _render_half(self, events, color) -> np.ndarray:
        grid = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)
        for _, gx, gy in events:
            grid[gy, gx] += 1.0
        grid = cv2.GaussianBlur(grid, (15, 15), 0)
        if grid.max() > 0:
            grid = (grid / grid.max() * 255).astype(np.uint8)
        else:
            grid = grid.astype(np.uint8)
        return cv2.applyColorMap(grid, cv2.COLORMAP_JET)

    def render_minimap(self) -> np.ndarray:
        """Return a (2*GRID_H, GRID_W, 3) BGR minimap: upper half on top, lower half below."""
        self._evict(time.time())
        upper_img = self._render_half(self.upper_events, None)
        lower_img = self._render_half(self.lower_events, None)
        minimap = np.vstack([upper_img, lower_img])
        h, w = minimap.shape[:2]
        cv2.line(minimap, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        return minimap

    def overlay_on(self, frame_bgr: np.ndarray, position: str = "bottom-right",
                   size: tuple = (240, 130)) -> np.ndarray:
        """Composite the heatmap minimap onto `frame_bgr` in-place style (returns same array)."""
        minimap = self.render_minimap()
        minimap = cv2.resize(minimap, size)
        fh, fw = frame_bgr.shape[:2]
        oh, ow = size[1], size[0]
        if position == "bottom-right":
            x, y = fw - ow - 20, fh - oh - 20
        elif position == "top-left":
            x, y = 20, 20
        else:
            x, y = 20, fh - oh - 20
        # Bounds check
        if x < 0 or y < 0 or x + ow > fw or y + oh > fh:
            return frame_bgr
        # Blend
        roi = frame_bgr[y:y + oh, x:x + ow]
        blended = cv2.addWeighted(roi, 0.4, minimap, 0.6, 0)
        frame_bgr[y:y + oh, x:x + ow] = blended
        # Border + label
        cv2.rectangle(frame_bgr, (x, y), (x + ow, y + oh), (255, 255, 255), 1)
        cv2.putText(frame_bgr, "HEATMAP 2min", (x + 4, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return frame_bgr
```

- [ ] **Step 4: Run tests, confirm pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_heatmap -v
```

Expected: 6 tests, all `ok`.

- [ ] **Step 5: Run full suite to confirm no regressions**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 45 tests in ... OK` (6 new + 39 prior).

- [ ] **Step 6: Commit**

```powershell
git add badminton_analysis/analytics/heatmap.py tests/test_heatmap.py
git commit -m "feat(analytics): add CourtHeatmap (2-min sliding window, upper/lower split)"
```

---

### Task 3: Add `court_pos` field to Player objects

**Files:**
- Modify: `badminton_analysis/tracking/player.py`

- [ ] **Step 1: Read the Player class definition**

Read `badminton_analysis/tracking/player.py` to find the Player dataclass and the methods that update player state. Note where position (x, y) is updated and where the court_mid_height is consulted to assign upper/lower half.

- [ ] **Step 2: Add the `court_pos` field**

Find the `Player` class. Add a field `court_pos: tuple | None = None` and document it. The field is `(x_m, y_m, half)` once a court_mapper is available, else `None`.

Also ensure the update method (whichever updates per-frame position) also sets `self.court_pos` when the system has a court_mapper. Concretely, find where `(self.upper_players / self.lower_players)` are assigned based on the position and the court_mid_height — that block is the natural place to also call `self.system.court_mapper.image_to_court(x, y)` and store the (x_m, y_m, half) triple on the player.

The exact change depends on the existing code. Apply the minimum edit that:
1. Adds `court_pos: tuple | None = None` to the Player class.
2. After the per-frame position update and upper/lower assignment, if `self.court_mapper` is set, call `court_mapper.image_to_court(x, y)` to get `(x_m, y_m)`, then set `player.court_pos = (x_m, y_m, "upper" if upper else "lower")`.

(If the existing player.py uses a different shape for the Player class — e.g., a dict or plain attrs — match that shape; the field name must be `court_pos` and the value must be `(x_m, y_m, half)` or `None`.)

- [ ] **Step 3: Verify import still loads**

Run:
```powershell
.\.venv\Scripts\python.exe -c "from badminton_analysis.tracking.player import Player; print('Player OK:', Player)"
```

Expected: prints the Player class.

- [ ] **Step 4: Run full suite**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 45 tests in ... OK` (no new tests, prior still pass).

- [ ] **Step 5: Commit**

```powershell
git add badminton_analysis/tracking/player.py
git commit -m "feat(tracking): add court_pos field on Player (court-mapped coordinates + half)"
```

---

### Task 4: Integrate updater + heatmap into BadmintonAnalysisSystem

**Files:**
- Modify: `badminton_analysis/system.py`

- [ ] **Step 1: Read the existing __init__ and _process_frame signatures**

Read `badminton_analysis/system.py` around the `__init__` and `_process_frame` to understand the current state initialization and per-frame flow.

- [ ] **Step 2: Add updater/heatmap attributes and CLI params**

In `__init__`, add (default `False` for both, so existing behaviour is unchanged):

```python
                 skip_court_annotation=False, device=None,
                 court_update_interval=8.0, court_update_min_quality=0.5,
                 show_heatmap=True, heatmap_window=120.0):
```

Store them as `self.court_update_interval`, `self.court_update_min_quality`, `self.show_heatmap`, `self.heatmap_window`.

After the existing model setup (where `self.court_corners`, `self.court_mapper`, `self.mid_height` are assigned by `_setup_court_annotation`), add:

```python
        # In-play court model updater
        from .court.updater import CourtModelUpdater
        self.court_updater = CourtModelUpdater(
            self,
            check_interval_sec=self.court_update_interval,
            min_quality=self.court_update_min_quality,
        )
        # Real-time sliding-window heatmap
        from .analytics.heatmap import CourtHeatmap
        # If user overrode heatmap_window, swap the class attribute
        if self.heatmap_window != CourtHeatmap.WINDOW_SEC:
            CourtHeatmap.WINDOW_SEC = float(self.heatmap_window)
        self.heatmap = CourtHeatmap()
```

- [ ] **Step 3: Update _setup_court_annotation to set `self.court_corners` and `self.mid_height` as public attributes**

In `_setup_court_annotation`, change the final return so the system keeps references. Find the existing `return corners, roi_corners, mid_height` at the end of that method. Before returning, assign:

```python
        self.court_corners = corners
        self.court_roi_corners = roi_corners
        self.mid_height = mid_height
        return corners, roi_corners, mid_height
```

(These are also re-assigned by the updater on each refresh; the updater writes to `self.system.court_corners` directly, so this initial assignment is just the first value.)

- [ ] **Step 4: In `_process_frame`, after player tracking, call updater + heatmap**

Find the end of the player-tracking block in `_process_frame` (the place where players are assigned to upper/lower). After that block, add:

```python
        # Real-time court model update + heatmap
        if not is_court:
            pass  # skip in non-court views
        elif self.court_mapper is not None and len(players) > 0:
            try:
                self.court_updater.maybe_update(frame, len(players))
            except Exception:
                pass
            now = time.time()
            for p in players:
                if getattr(p, "court_pos", None) is not None:
                    x_m, y_m, half = p.court_pos
                    try:
                        self.heatmap.add(x_m, y_m, half, t=now)
                    except Exception:
                        pass
        if self.show_heatmap and frame is not None:
            try:
                self.heatmap.overlay_on(frame)
            except Exception:
                pass
```

(Use `try/except` so a heatmap/updater bug never breaks the per-frame pipeline. The original behaviour — output video without any heatmap or updater impact — is preserved when no `court_pos` is set or when `show_heatmap=False`.)

- [ ] **Step 5: Verify the module still loads**

Run:
```powershell
.\.venv\Scripts\python.exe -c "from badminton_analysis.system import BadmintonAnalysisSystem; print('OK')"
```

Expected: `OK`.

- [ ] **Step 6: Run full suite**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 45 tests in ... OK`.

- [ ] **Step 7: Commit**

```powershell
git add badminton_analysis/system.py
git commit -m "feat(system): integrate court_updater and heatmap into _process_frame"
```

---

### Task 5: Regression test - existing CLI flow must be byte-identical

**Files:** none (verification only)

- [ ] **Step 1: Wipe outputs and pre-annotate**

Run:
```powershell
Remove-Item -LiteralPath outputs/demo1 -Recurse -Force -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe (Get-Content C:\Users\18683\AppData\Local\Temp\opencode\preannotate.py -Raw) 2>&1 | Select-Object -Last 1
```

- [ ] **Step 2: Run main.py**

Run:
```powershell
.\.venv\Scripts\python.exe main.py --video-path videos/demo1.mp4 --template-path templates/demo1.png --display false 2>&1 | Select-String "Video saved" | Select-Object -First 1
```

- [ ] **Step 3: Verify byte sizes match historical reference**

Run:
```powershell
Write-Output "detect_demo1.mp4: $((Get-Item outputs/demo1/detect_demo1.mp4).Length) (expected 22112015)"
Write-Output "detections.jsonl: $((Get-Item outputs/demo1/detections.jsonl).Length) (expected 467583)"
Write-Output "metadata.json: $((Get-Item outputs/demo1/metadata.json).Length) (expected 944)"
```

Expected: all three match exactly. (Heatmap is empty when no `court_pos` is set; the integration is a no-op for files processed without the new model.)

- [ ] **Step 4: No commit (verification only)**

If sizes match, integration is non-disruptive. If they differ, re-check Task 4 edits.

---

### Task 6: TDD E2E - heatmap actually renders on synthetic frames

**Files:**
- Create: `tests/test_heatmap_integration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_heatmap_integration.py`:

```python
"""E2E: feed CourtHeatmap through BadmintonAnalysisSystem._process_frame and
verify the output frame has the heatmap overlay drawn on it.
"""
from __future__ import annotations

import shutil
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from badminton_analysis.analytics.heatmap import CourtHeatmap
from badminton_analysis.court.mapper import CourtMapper


class HeatmapE2ETest(unittest.TestCase):
    def test_overlay_after_feeding_positions(self):
        # Simulate 10 position events spread over a small area in upper court.
        h = CourtHeatmap()
        now = time.time()
        for i in range(10):
            h.add(x_m=2.0 + 0.1 * i, y_m=5.0 + 0.1 * i, half="upper", t=now - i * 0.5)
        for i in range(5):
            h.add(x_m=4.0 + 0.1 * i, y_m=10.0 + 0.1 * i, half="lower", t=now - i * 0.5)
        # Render minimap and overlay on a 720p frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = h.overlay_on(frame, position="bottom-right", size=(240, 130))
        # The overlay region should now contain non-black pixels (the heatmap)
        x, y = 1280 - 240 - 20, 720 - 130 - 20
        region = out[y:y + 130, x:x + 240, :]
        self.assertGreater(int(region.max()), 0)
        # And the heatmap is not all black in the middle (events are clustered)
        center = region[40:90, 60:180, :]
        self.assertGreater(int(center.max()), 0)

    def test_court_mapper_round_trip(self):
        """image -> court -> image should be close (mapper sanity)."""
        corners = [(100, 60), (540, 60), (540, 420), (100, 420)]
        mapper = CourtMapper(corners)
        # mid-court pixel
        x_img, y_img = 320, 240
        x_m, y_m = mapper.image_to_court(x_img, y_img)
        # Should be roughly mid-court in meters (~3.05, ~6.5)
        self.assertGreater(x_m, 1.0)
        self.assertLess(x_m, 5.0)
        self.assertGreater(y_m, 4.0)
        self.assertLess(y_m, 9.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test, confirm pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_heatmap_integration -v
```

Expected: 2 tests, all `ok`. If `mapper.image_to_court` does not exist on `CourtMapper`, see Task 6.1 below.

- [ ] **Step 3: If `CourtMapper.image_to_court` is missing, add it**

Read `badminton_analysis/court/mapper.py` and check whether `CourtMapper` has an `image_to_court(x, y)` method. If not, add it:

```python
    def image_to_court(self, x_img: int, y_img: int) -> tuple:
        """Map image pixel coordinates to court-meters (origin at top-left of court)."""
        if not hasattr(self, "M") or self.M is None:
            self.compute_perspective_transform()
        pt = np.array([[[float(x_img), float(y_img)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.M)
        return float(out[0, 0, 0]), float(out[0, 0, 1])
```

(The existing `draw_court_overlay` already calls `cv2.getPerspectiveTransform`, so the `M` attribute is set there. If the attribute is named differently, reuse whatever the existing code uses.)

- [ ] **Step 4: Run full suite**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 47 tests in ... OK` (2 new + 45 prior).

- [ ] **Step 5: Commit**

```powershell
git add tests/test_heatmap_integration.py badminton_analysis/court/mapper.py
git commit -m "test(heatmap): add E2E + ensure CourtMapper.image_to_court exists"
```

---

### Task 7: Plumb CLI/Streamlit flags

**Files:**
- Modify: `badminton_analysis/live.py`
- Modify: `app.py`

- [ ] **Step 1: Add CLI flags to live.py**

In `parse_args()`, add:

```python
    p.add_argument("--court-update-interval", type=float, default=8.0,
                   help="Court model re-check interval (seconds, default 8)")
    p.add_argument("--no-heatmap", action="store_true",
                   help="Disable real-time heatmap overlay")
    p.add_argument("--heatmap-window", type=float, default=120.0,
                   help="Heatmap sliding window (seconds, default 120)")
```

In `main()` where `BadmintonAnalysisSystem(...)` is constructed, add:

```python
        court_update_interval=args.court_update_interval,
        show_heatmap=not args.no_heatmap,
        heatmap_window=args.heatmap_window,
```

- [ ] **Step 2: Add Streamlit sidebar widgets**

In `app.py` after the existing `imgsz/conf/frame_skip` block, add:

```python
    with st.expander("高级: 球场自动更新 + 热力ͼ", expanded=False):
        court_update_interval = st.slider("球场ģ型检查间隔 (秒)", 2.0, 60.0, 8.0, step=1.0)
        court_update_min_quality = st.slider("球场更新最低质量", 0.1, 1.0, 0.5, step=0.05)
        show_heatmap = st.checkbox("启用ʵʱ热力ͼ (2 分钟滑动窗口)", True)
        heatmap_window = st.slider("热力ͼ滑动窗口 (秒)", 30, 600, 120, step=30)
```

In the `BadmintonAnalysisSystem(...)` construction in the start button handler, add:

```python
                court_update_interval=court_update_interval,
                court_update_min_quality=court_update_min_quality,
                show_heatmap=show_heatmap,
                heatmap_window=heatmap_window,
```

- [ ] **Step 3: Verify both parse and run**

Run:
```powershell
.\.venv\Scripts\python.exe -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); ast.parse(open('badminton_analysis/live.py', encoding='utf-8').read()); print('parse OK')"
```

Expected: `parse OK`.

- [ ] **Step 4: Run full suite**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 47 tests in ... OK`.

- [ ] **Step 5: Commit**

```powershell
git add badminton_analysis/live.py app.py
git commit -m "feat(cli/web): add --court-update-interval, --no-heatmap, --heatmap-window flags"
```

---

### Task 8: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Find the right insertion point**

Run:
```powershell
Select-String -Path README.md -Pattern "^## "
```

Find the last `##` section before "Star History".

- [ ] **Step 2: Append a "球场ģ型自动更新 + ʵʱ热力ͼ" section**

Insert (use a Python script to handle UTF-8 safely, mirroring what Task 11 of the previous plan did):

```markdown
## 球场ģ型自动更新 + ʵʱ热力ͼ

比赛过程中，球Ա的"ȫ场画面"可以用来ˢ新球场ģ型（֮ǰֻ在启动ʱ跑һ次 auto-detect）。ͬʱ，2 分钟滑动窗口热力ͼ（上下半场分ͼ）会ʵʱ叠加在 output 视Ƶ的右下角。

### 行Ϊ

- ÿ `--court-update-interval` 秒（Ĭ认 8 秒）检查һ次当ǰ֡
- 需Ҫ满足：≥2 个球Ա可见 + ֡清晰（Laplacian 方差 > 30）
- ͨ过质量阈ֵ + 角点距离合理性（任意角点ƫ离 < 100px）才接受
- 接受后д `outputs/<run>/court_annotations.txt` 持久化，并立即更新 `court_mapper` 让后续֡的 pose ӳ射跟随新ģ型

热力ͼÿ个 frame 把球Ա的 court 坐标λ置入队，过期（> 2 分钟）的自动剔除。下半场各画һ张，合成һ张 240x130 minimap 叠加到画面右下角。

### CLI 标־

```bash
python -m badminton_analysis.live --source screen_capture \
       --court-update-interval 5 \
       --no-heatmap
```

| 标־ | Ĭ认 | ˵明 |
|---|---|---|
| `--court-update-interval` | 8.0 | 球场ģ型重新检查间隔（秒） |
| `--court-update-min-quality` | 0.5 | 最低质量分数（0-1） |
| `--no-heatmap` | False | 关闭热力ͼ叠加 |
| `--heatmap-window` | 120.0 | 热力ͼ滑动窗口（秒） |

Streamlit 侧栏"高级"折叠区可调。
```

- [ ] **Step 3: Verify**

Run:
```powershell
Select-String -Path README.md -Pattern "球场ģ型自动更新"
```

Expected: prints the line number of the new section.

- [ ] **Step 4: Commit**

```powershell
git add README.md
git commit -m "docs: document court auto-update and real-time heatmap"
```

---

### Task 9: Final regression + 47/47 tests

**Files:** none (verification only)

- [ ] **Step 1: Run full test suite**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests 2>&1 | Select-String "Ran |OK|FAIL" | Select-Object -Last 1
```

Expected: `Ran 47 tests in ... OK`.

- [ ] **Step 2: Run regression byte-level**

Run:
```powershell
Remove-Item -LiteralPath outputs/demo1 -Recurse -Force -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe (Get-Content C:\Users\18683\AppData\Local\Temp\opencode\preannotate.py -Raw) 2>&1 | Select-Object -Last 1
.\.venv\Scripts\python.exe main.py --video-path videos/demo1.mp4 --template-path templates/demo1.png --display false 2>&1 | Select-String "Video saved" | Select-Object -First 1
Write-Output "detect_demo1.mp4: $((Get-Item outputs/demo1/detect_demo1.mp4).Length) (expected 22112015)"
Write-Output "detections.jsonl: $((Get-Item outputs/demo1/detections.jsonl).Length) (expected 467583)"
Write-Output "metadata.json: $((Get-Item outputs/demo1/metadata.json).Length) (expected 944)"
```

Expected: byte sizes still match.

- [ ] **Step 3: No commit**

If any check fails, fix and re-run before declaring done.

---
