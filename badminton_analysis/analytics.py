"""Analytics primitives: FPS, ball trail, per-frame stats, multi-frame aggregation."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
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
        if self._fps == 0.0:
            self._fps = instant
        else:
            self._fps = self._alpha * self._fps + (1 - self._alpha) * instant
        return self._fps


class BallTrail:
    """Rolling buffer of the ball's recent centers (x, y)."""

    def __init__(self, maxlen: int = 30):
        self._buf: deque = deque(maxlen=maxlen)

    def update(self, tracks: list) -> list:
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
