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
    cv2.line(img, (100, 100), (w - 100, h - 100), (255, 255, 255), 2)
    return img


def _blurry_frame(h=720, w=1280):
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
        self.assertEqual(u._score_quality(_sharp_frame(), player_count=0), 0.0)
        self.assertEqual(u._score_quality(_sharp_frame(), player_count=1), 0.0)

    def test_quality_low_when_blurry(self):
        u = CourtModelUpdater(self._make_system(), check_interval_sec=0)
        self.assertEqual(u._score_quality(_blurry_frame(), player_count=2), 0.0)

    def test_quality_high_with_2_players_and_sharp_frame(self):
        u = CourtModelUpdater(self._make_system(), check_interval_sec=0)
        q = u._score_quality(_sharp_frame(), player_count=2)
        self.assertGreater(q, 0.3)
        self.assertLessEqual(q, 1.0)

    def test_should_check_respects_interval(self):
        system = self._make_system(current_corners=[(0, 0), (640, 0), (640, 480), (0, 480)])
        u = CourtModelUpdater(system, check_interval_sec=10.0)
        self.assertFalse(u._should_check())
        u._last_check = time.time() - 11.0
        self.assertTrue(u._should_check())

    def test_is_plausible_update_no_prior_corners(self):
        u = CourtModelUpdater(self._make_system(current_corners=None))
        self.assertTrue(u._is_plausible_update([(0, 0), (100, 0), (100, 100), (0, 100)]))

    def test_is_plausible_update_rejects_huge_jump(self):
        system = self._make_system(current_corners=[(100, 100), (200, 100), (200, 200), (100, 200)])
        u = CourtModelUpdater(system, check_interval_sec=0)
        self.assertFalse(u._is_plausible_update([(0, 0), (100, 0), (100, 100), (0, 100)]))

    def test_is_plausible_update_accepts_small_drift(self):
        system = self._make_system(current_corners=[(100, 100), (200, 100), (200, 200), (100, 200)])
        u = CourtModelUpdater(system, check_interval_sec=0)
        self.assertTrue(u._is_plausible_update([(120, 100), (200, 100), (200, 200), (100, 200)]))


if __name__ == "__main__":
    unittest.main()
