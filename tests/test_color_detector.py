"""Unit tests for color-based court detection."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from badminton_analysis.court.detector import (
    auto_detect_court_corners,
    detect_court_by_color,
)


def _synthetic_red_court(h=720, w=1280, margin=40):
    """A solid red rectangle in the middle of a black image (no perspective).

    margin=40 gives a 600x640 red block (~56% of frame), comfortably above
    the 20% area threshold.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # BGR red = (40, 40, 200) for BWF-like saturation
    img[margin:h - margin, margin:w - margin] = (40, 40, 200)
    return img


def _synthetic_red_court_perspective(h=720, w=1280):
    """A trapezoid red court: narrower at the top (perspective)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    top_y = int(h * 0.2)
    bot_y = int(h * 0.95)
    top_lx = int(w * 0.3)
    top_rx = int(w * 0.7)
    bot_lx = int(w * 0.1)
    bot_rx = int(w * 0.9)
    pts = np.array([[top_lx, top_y], [top_rx, top_y],
                   [bot_rx, bot_y], [bot_lx, bot_y]], dtype=np.int32)
    cv2.fillPoly(img, [pts], color=(40, 40, 200))
    return img


def _synthetic_green_court(h=720, w=1280, margin=40):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[margin:h - margin, margin:w - margin] = (40, 180, 40)
    return img


class ColorDetectorTest(unittest.TestCase):
    def test_detect_red_rectangle_returns_4_corners(self):
        img = _synthetic_red_court()
        corners = detect_court_by_color(img, color="red")
        self.assertIsNotNone(corners)
        self.assertEqual(len(corners), 4)
        for x, y in corners:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)

    def test_red_corners_are_at_red_boundary(self):
        img = _synthetic_red_court(h=720, w=1280, margin=40)
        corners = detect_court_by_color(img, color="red")
        self.assertIsNotNone(corners)
        # The red rect spans (40, 40) -> (1240, 680).
        xs = sorted([p[0] for p in corners])
        ys = sorted([p[1] for p in corners])
        # Allow +-10px tolerance
        self.assertAlmostEqual(xs[0], 40, delta=10)
        self.assertAlmostEqual(xs[-1], 1240, delta=10)
        self.assertAlmostEqual(ys[0], 40, delta=10)
        self.assertAlmostEqual(ys[-1], 680, delta=10)

    def test_perspective_red_court(self):
        img = _synthetic_red_court_perspective()
        corners = detect_court_by_color(img, color="red")
        self.assertIsNotNone(corners)
        self.assertEqual(len(corners), 4)

    def test_no_red_returns_none(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)  # all black
        corners = detect_court_by_color(img, color="red")
        self.assertIsNone(corners)

    def test_green_court(self):
        img = _synthetic_green_court()
        corners = detect_court_by_color(img, color="green")
        self.assertIsNotNone(corners)
        self.assertEqual(len(corners), 4)

    def test_auto_detect_uses_color_first(self):
        img = _synthetic_red_court()
        corners, mask, debug = auto_detect_court_corners(img)
        self.assertIsNotNone(corners)
        self.assertEqual(debug.get("method"), "color")
        self.assertEqual(debug.get("color"), "red")
        self.assertIsNone(mask)  # color path doesn't return a mask

    def test_auto_detect_falls_back_to_line_for_non_court_colors(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)  # black
        corners, mask, debug = auto_detect_court_corners(img)
        # Falls back to line detection; black image won't have lines either
        self.assertIsNone(corners)
        # debug indicates the fallback path
        self.assertIn("method", debug)


if __name__ == "__main__":
    unittest.main()
