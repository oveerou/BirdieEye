"""Unit tests for storage (SQLite + metrics.json)."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from badminton_analysis.storage import (
    RunMetrics,
    init_db,
    insert_run,
    list_runs,
    save_metrics_json,
)


def _sample_metrics():
    return RunMetrics(
        run_id="run_test_001",
        source_type="screen_capture",
        source_ref="100,100,640,480",
        model_path="weights/yolo11n-pose.pt",
        device="cuda:0",
        conf=0.25,
        imgsz=960,
        frame_skip=1,
        total_frames=300,
        avg_fps=28.5,
        avg_player_count=2.0,
        ball_visible_ratio=0.8,
        upper_avg_speed=0.4,
        lower_avg_speed=0.3,
        total_rallies=5,
    )


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = Path(self.tmp) / "test.db"
        self.run_dir = Path(self.tmp) / "run_test_001"

    def test_init_db_creates_table(self):
        conn = init_db(self.db_path)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
        )
        self.assertIsNotNone(cur.fetchone())
        conn.close()

    def test_insert_and_list_run(self):
        conn = init_db(self.db_path)
        insert_run(conn, _sample_metrics())
        conn.close()
        conn = init_db(self.db_path)
        rows = list_runs(conn, limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "run_test_001")
        self.assertEqual(rows[0]["source_type"], "screen_capture")
        self.assertEqual(rows[0]["total_rallies"], 5)
        conn.close()

    def test_save_metrics_json(self):
        m = _sample_metrics()
        save_metrics_json(m, self.run_dir)
        out = self.run_dir / "metrics.json"
        self.assertTrue(out.exists())
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["run_id"], "run_test_001")
        self.assertEqual(data["upper_avg_speed"], 0.4)
        self.assertEqual(data["total_rallies"], 5)

    def test_runmetrics_required_fields(self):
        m = _sample_metrics()
        self.assertEqual(m.run_id, "run_test_001")
        d = m.to_dict()
        for k in ("run_id", "source_type", "source_ref", "model_path", "device",
                  "conf", "imgsz", "frame_skip", "total_frames", "avg_fps",
                  "avg_player_count", "ball_visible_ratio", "upper_avg_speed",
                  "lower_avg_speed", "total_rallies"):
            self.assertIn(k, d)


if __name__ == "__main__":
    unittest.main()
