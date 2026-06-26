"""End-to-end: spawn a thread that calls process_video, push frames through
display_queue, consume from the main thread, verify frames flow end-to-end.
"""
from __future__ import annotations

import queue
import shutil
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from badminton_analysis.sources.stream_adapter import StreamAdapter
from tests.fake_source import FakeFrameSource


class WebFlowE2ETest(unittest.TestCase):
    def test_threaded_flow_pushes_frames_to_queue(self):
        from badminton_analysis.system import BadmintonAnalysisSystem, load_runtime_dependencies
        load_runtime_dependencies()

        frames = [np.full((480, 640, 3), (i * 10) % 256, dtype=np.uint8) for i in range(20)]
        first = frames[0]
        save_dir = Path("outputs").resolve() / "_web_e2e_test"
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True)
        template_path = save_dir / "first_frame.png"
        cv2.imwrite(str(template_path), first)
        with open(save_dir / "court_annotations.txt", "w") as f:
            f.write("corners=[(80, 60), (560, 60), (560, 420), (80, 420)]\n")
            f.write("roi_corners=[(60, 0), (580, 480)]\n")
            f.write("mid_height=240\n")

        fake = FakeFrameSource(frames=frames)
        adapter = StreamAdapter(source=fake, fps=30.0, first_frame=first)
        display_q: queue.Queue = queue.Queue(maxsize=4)
        stop = threading.Event()

        system = BadmintonAnalysisSystem(
            video_path="_web_e2e.mp4",
            template_path=str(template_path),
            output_dir=str(save_dir),
            ball_model_path="weights/yolo11s-ball.pt",
            pose_family="yolo-pose",
            yolo_pose_model="weights/yolo11n-pose.pt",
            show_display=False,
            show_skeletons=False,
            show_player_trajectories=False,
            show_court_trajectory=False,
            show_shuttlecock_trajectory=False,
            show_player_stats=False,
            frame_source=adapter,
            non_interactive_annotation=True,
        )
        system.keep_audio = False
        system.display_queue = display_q

        def worker():
            try:
                system.process_video()
            except Exception as e:
                print(f"[e2e worker] error: {e}")
            finally:
                stop.set()
                adapter.release()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        consumed = 0
        deadline = time.time() + 30
        while consumed < 3 and time.time() < deadline:
            try:
                frame, idx = display_q.get(timeout=2.0)
                self.assertIsNotNone(frame)
                self.assertEqual(frame.shape[2], 3)
                consumed += 1
            except queue.Empty:
                if stop.is_set():
                    break
        stop.set()
        t.join(timeout=10)

        self.assertGreater(consumed, 0, "expected at least one frame from the queue")


if __name__ == "__main__":
    unittest.main()
