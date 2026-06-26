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
        self.assertTrue(str(c.outputs_dir).endswith("outputs"))
        self.assertTrue(str(c.runs_dir).replace("\\", "/").endswith("outputs/runs"))
        self.assertTrue(str(c.uploads_dir).endswith("uploads"))

    def test_load_config_returns_appconfig(self):
        c = load_config()
        self.assertIsInstance(c, AppConfig)


if __name__ == "__main__":
    unittest.main()
