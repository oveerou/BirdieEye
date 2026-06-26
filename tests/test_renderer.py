"""Unit tests for renderer.render_frame."""
from __future__ import annotations

import unittest

import numpy as np

from badminton_analysis.analytics import FrameStats
from badminton_analysis.renderer import render_frame


def _blank(h=480, w=640, color=(0, 0, 0)):
    return np.full((h, w, 3), color, dtype=np.uint8)


class RenderFrameTest(unittest.TestCase):
    def test_returns_bgr_frame_same_shape(self):
        frame = _blank()
        stats = FrameStats(
            frame_idx=0, fps=30.0, player_count=0,
            ball_visible=False, ball_center=None, rally_id=0,
            upper_player_count=0, lower_player_count=0,
            avg_speed_upper=0.0, avg_speed_lower=0.0,
            total_distance_upper=0.0, total_distance_lower=0.0,
        )
        out = render_frame(frame, [], [], stats, device="cuda:0")
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_draws_status_bar(self):
        frame = _blank(720, 1280)
        stats = FrameStats(
            frame_idx=42, fps=29.7, player_count=2,
            ball_visible=True, ball_center=(100, 200), rally_id=3,
            upper_player_count=1, lower_player_count=1,
            avg_speed_upper=0.5, avg_speed_lower=0.3,
            total_distance_upper=10.0, total_distance_lower=5.0,
        )
        out = render_frame(frame, [], [], stats, device="cuda:0")
        top_strip = out[0:60, :, :]
        self.assertGreater(int(top_strip.max()), 0)

    def test_draws_ball_trail(self):
        frame = _blank(720, 1280)
        trail = [(100, 100), (200, 200), (300, 300)]
        stats = FrameStats(
            frame_idx=1, fps=30.0, player_count=0,
            ball_visible=True, ball_center=(300, 300), rally_id=0,
            upper_player_count=0, lower_player_count=0,
            avg_speed_upper=0.0, avg_speed_lower=0.0,
            total_distance_upper=0.0, total_distance_lower=0.0,
        )
        out = render_frame(frame, [], trail, stats, device="cuda:0")
        self.assertGreater(int(out[200, 200, :].max()), 0)

    def test_no_trail_no_panic(self):
        frame = _blank()
        stats = FrameStats(
            frame_idx=0, fps=0.0, player_count=0,
            ball_visible=False, ball_center=None, rally_id=0,
            upper_player_count=0, lower_player_count=0,
            avg_speed_upper=0.0, avg_speed_lower=0.0,
            total_distance_upper=0.0, total_distance_lower=0.0,
        )
        out = render_frame(frame, [], [], stats, device="cpu")
        self.assertEqual(out.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
