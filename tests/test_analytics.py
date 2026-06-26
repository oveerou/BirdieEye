"""Unit tests for analytics primitives."""
from __future__ import annotations

import unittest

from badminton_analysis.analytics import (
    BallTrail,
    FpsCounter,
    FrameStats,
    MetricsAggregator,
)


class FpsCounterTest(unittest.TestCase):
    def test_first_tick_returns_zero(self):
        f = FpsCounter(alpha=0.9)
        self.assertEqual(f.tick(), 0.0)

    def test_subsequent_ticks_nonneg(self):
        f = FpsCounter(alpha=0.5)
        f.tick()
        v = f.tick()
        self.assertGreaterEqual(v, 0.0)


class BallTrailTest(unittest.TestCase):
    def test_empty_initially(self):
        t = BallTrail(maxlen=10)
        self.assertEqual(t.points(), [])

    def test_update_appends_centers(self):
        t = BallTrail(maxlen=10)

        class FakeTrack:
            def __init__(self, x, y):
                self.x, self.y = x, y
        t.update([FakeTrack(10, 20)])
        self.assertEqual(t.points(), [(10, 20)])

    def test_maxlen_caps_history(self):
        t = BallTrail(maxlen=2)

        class P:
            def __init__(self, x, y):
                self.x, self.y = x, y
        t.update([P(1, 1)])
        t.update([P(2, 2)])
        t.update([P(3, 3)])
        self.assertEqual(len(t.points()), 2)
        self.assertEqual(t.points()[-1], (3, 3))


class FrameStatsTest(unittest.TestCase):
    def test_default_fields(self):
        s = FrameStats(
            frame_idx=10, fps=30.0, player_count=2,
            ball_visible=True, ball_center=(100, 200), rally_id=1,
            upper_player_count=1, lower_player_count=1,
            avg_speed_upper=0.5, avg_speed_lower=0.3,
            total_distance_upper=10.0, total_distance_lower=7.0,
        )
        self.assertEqual(s.frame_idx, 10)
        self.assertTrue(s.ball_visible)
        self.assertEqual(s.upper_player_count, 1)


class MetricsAggregatorTest(unittest.TestCase):
    def _stat(self, **kw):
        defaults = dict(
            frame_idx=0, fps=30.0, player_count=2,
            ball_visible=True, ball_center=(0, 0), rally_id=0,
            upper_player_count=1, lower_player_count=1,
            avg_speed_upper=0.0, avg_speed_lower=0.0,
            total_distance_upper=0.0, total_distance_lower=0.0,
        )
        defaults.update(kw)
        return FrameStats(**defaults)

    def test_empty_summary(self):
        agg = MetricsAggregator()
        s = agg.summary()
        self.assertEqual(s["total_frames"], 0)
        self.assertEqual(s["avg_fps"], 0.0)
        self.assertEqual(s["total_rallies"], 0)

    def test_aggregates_averages(self):
        agg = MetricsAggregator()
        agg.update(self._stat(fps=20.0, player_count=2, ball_visible=True,
                              upper_player_count=1, lower_player_count=1,
                              avg_speed_upper=0.5, avg_speed_lower=0.3,
                              total_distance_upper=10.0, total_distance_lower=5.0))
        agg.update(self._stat(fps=40.0, player_count=4, ball_visible=False,
                              upper_player_count=2, lower_player_count=2,
                              avg_speed_upper=1.0, avg_speed_lower=0.6,
                              total_distance_upper=20.0, total_distance_lower=15.0))
        s = agg.summary()
        self.assertEqual(s["total_frames"], 2)
        self.assertAlmostEqual(s["avg_fps"], 30.0)
        self.assertEqual(s["ball_visible_count"], 1)
        self.assertAlmostEqual(s["ball_visible_ratio"], 0.5)
        self.assertEqual(s["total_rallies"], 0)

    def test_tracks_rallies(self):
        agg = MetricsAggregator()
        for rid in (1, 1, 2, 2, 2):
            agg.update(self._stat(rally_id=rid))
        s = agg.summary()
        self.assertEqual(s["total_rallies"], 2)


if __name__ == "__main__":
    unittest.main()
