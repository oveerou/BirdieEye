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
        h.add(3.05, 6.5, "upper", t=1.0)
        self.assertEqual(len(h.upper_events), 1)
        ts, gx, gy = h.upper_events[0]
        self.assertEqual(ts, 1.0)
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
        h.add(3.0, 6.0, "upper", t=100.0 + 200.0)
        h._evict(now=100.0 + 200.0)
        self.assertEqual(len(h.upper_events), 1)

    def test_render_minimap_shape(self):
        h = CourtHeatmap()
        h.add(3.0, 6.0, "upper", t=1.0)
        h.add(3.0, 11.0, "lower", t=1.0)
        minimap = h.render_minimap()
        self.assertEqual(minimap.shape, (CourtHeatmap.GRID_H * 2, CourtHeatmap.GRID_W, 3))
        self.assertEqual(minimap.dtype, np.uint8)

    def test_overlay_on_returns_same_shape(self):
        h = CourtHeatmap()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        h.add(3.0, 6.0, "upper", t=1.0)
        out = h.overlay_on(frame, position="bottom-right", size=(240, 130))
        self.assertEqual(out.shape, frame.shape)
        region = out[720 - 130 - 20:720 - 20, 1280 - 240 - 20:1280 - 20, :]
        self.assertGreater(int(region.max()), 0)


if __name__ == "__main__":
    unittest.main()
