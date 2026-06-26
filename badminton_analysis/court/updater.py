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

    MAX_CORNER_JUMP_PX = 100

    def __init__(self, system, check_interval_sec: float = 8.0, min_quality: float = 0.5):
        self.system = system
        self.check_interval_sec = float(check_interval_sec)
        self.min_quality = float(min_quality)
        self._last_check = time.time()
        self._update_count = 0
        self._last_quality = 0.0
        self._last_status = "init"

    def _should_check(self) -> bool:
        return (time.time() - self._last_check) >= self.check_interval_sec

    def _score_quality(self, frame, player_count: int) -> float:
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
        prior = self.system.court_corners
        if not prior:
            return True
        for old, new in zip(prior, new_corners):
            dist = float(((old[0] - new[0]) ** 2 + (old[1] - new[1]) ** 2) ** 0.5)
            if dist > self.MAX_CORNER_JUMP_PX:
                return False
        return True

    def _detect(self, frame):
        h, w = frame.shape[:2]
        fixed = (1080, 720)
        base = cv2.resize(frame, fixed)
        corners, _line, _dbg = auto_detect_court_corners(base)
        if not corners:
            return None, None, None
        roi = compute_expanded_roi(corners, base.shape)
        mapper = CourtMapper(corners)
        _, mid = mapper.draw_court_overlay(base)
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
        self.system.court_corners = new_corners
        self.system.court_roi_corners = new_roi
        self.system.mid_height = new_mid
        self.system.court_mapper = CourtMapper(new_corners)
        self._update_count += 1
        self._last_status = f"updated (#{self._update_count})"
        self._save_annotations(new_corners, new_roi, new_mid)
        return True
