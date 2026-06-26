"""E2E: feed CourtHeatmap with positions and verify the output frame has
the heatmap overlay drawn on it. Also verify CourtMapper.image_to_court works.
"""
from __future__ import annotations

import time
import unittest

import cv2
import numpy as np

from badminton_analysis.analytics.heatmap import CourtHeatmap
from badminton_analysis.court.mapper import CourtMapper


class HeatmapE2ETest(unittest.TestCase):
    def test_overlay_after_feeding_positions(self):
        h = CourtHeatmap()
        now = time.time()
        for i in range(10):
            h.add(x_m=2.0 + 0.1 * i, y_m=5.0 + 0.1 * i, half="upper", t=now - i * 0.5)
        for i in range(5):
            h.add(x_m=4.0 + 0.1 * i, y_m=10.0 + 0.1 * i, half="lower", t=now - i * 0.5)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = h.overlay_on(frame, position="bottom-right", size=(240, 130))
        self.assertEqual(out.shape, frame.shape)
        x, y = 1280 - 240 - 20, 720 - 130 - 20
        region = out[y:y + 130, x:x + 240, :]
        self.assertGreater(int(region.max()), 0)
        center = region[40:90, 60:180, :]
        self.assertGreater(int(center.max()), 0)

    def test_court_mapper_round_trip(self):
        corners = [(100, 60), (540, 60), (540, 420), (100, 420)]
        mapper = CourtMapper(corners)
        centroid = (320, 240)
        result = mapper.image_to_court(centroid)
        # result is a tuple (x_m, y_m)
        x_m, y_m = result[0], result[1]
        self.assertGreater(x_m, 1.0)
        self.assertLess(x_m, 5.0)
        self.assertGreater(y_m, 4.0)
        self.assertLess(y_m, 9.0)


if __name__ == "__main__":
    unittest.main()
