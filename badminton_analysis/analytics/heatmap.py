"""Real-time sliding-window heatmap of player positions in court coordinates."""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


class CourtHeatmap:
    """Per-half sliding-window heatmap over a 6.1m x 13.4m standard court."""

    COURT_W_M = 6.1
    COURT_H_M = 13.4
    GRID_W = 60
    GRID_H = 130
    WINDOW_SEC = 120.0  # 2 minutes
    MAX_EVENTS = 36000  # safety cap: 30 fps * 120 s

    def __init__(self):
        self.upper_events: deque = deque(maxlen=self.MAX_EVENTS)
        self.lower_events: deque = deque(maxlen=self.MAX_EVENTS)

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

    def _render_half(self, events) -> np.ndarray:
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
        self._evict(time.time())
        upper_img = self._render_half(self.upper_events)
        lower_img = self._render_half(self.lower_events)
        minimap = np.vstack([upper_img, lower_img])
        h, w = minimap.shape[:2]
        cv2.line(minimap, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        return minimap

    def overlay_on(self, frame_bgr: np.ndarray, position: str = "bottom-right",
                   size: tuple = (240, 130)) -> np.ndarray:
        if not self.upper_events and not self.lower_events:
            return frame_bgr
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
        if x < 0 or y < 0 or x + ow > fw or y + oh > fh:
            return frame_bgr
        roi = frame_bgr[y:y + oh, x:x + ow]
        blended = cv2.addWeighted(roi, 0.4, minimap, 0.6, 0)
        frame_bgr[y:y + oh, x:x + ow] = blended
        cv2.rectangle(frame_bgr, (x, y), (x + ow, y + oh), (255, 255, 255), 1)
        cv2.putText(frame_bgr, "HEATMAP 2min", (x + 4, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return frame_bgr
