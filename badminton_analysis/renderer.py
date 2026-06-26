"""Single-frame overlay renderer.

Side-effect-free: takes a BGR frame and per-frame state, returns a BGR frame.
Called from `BadmintonAnalysisSystem._process_frame` (CLI/live.py path) and
the Streamlit app's per-frame loop (web UI path).
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

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
