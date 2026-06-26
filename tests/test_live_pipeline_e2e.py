"""End-to-end test: BadmintonAnalysisSystem with a StreamAdapter (no real source).

This validates that the new frame_source path works through the full pipeline,
without requiring a real screen, browser, or display window. Uses a fake
FrameSource that returns scripted synthetic frames.
"""
from __future__ import annotations

import os
import shutil
import sys
import unittest

import numpy as np

from badminton_analysis.sources.stream_adapter import StreamAdapter
from tests.fake_source import FakeFrameSource


def _make_synthetic_video(num_frames: int = 30, h: int = 480, w: int = 640) -> list[np.ndarray]:
    """Generate a sequence of simple frames (solid colors with slight noise)."""
    rng = np.random.default_rng(0)
    frames = []
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 8) % 256            # blue gradient
        frame[:, :, 1] = 64                        # fixed green
        frame[:, :, 2] = ((i * 4) + 128) % 256     # red gradient
        noise = (rng.integers(-5, 5, frame.shape)).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


class LivePipelineE2ETest(unittest.TestCase):
    def test_frame_source_path_completes(self):
        from badminton_analysis.system import BadmintonAnalysisSystem, load_runtime_dependencies
        load_runtime_dependencies()

        num_frames = 30
        frames = _make_synthetic_video(num_frames, h=480, w=640)
        first = frames[0]
        # Pre-create a "court template" image from the first frame so system can annotate it.
        save_dir = os.path.abspath("outputs/_e2e_test")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        template_path = os.path.join(save_dir, "first_frame.png")
        import cv2
        cv2.imwrite(template_path, first)
        # Pre-write a court_annotations.txt with 4 plausible corners so we skip the GUI dialog
        corners = [(100, 100), (540, 100), (540, 380), (100, 380)]
        roi_corners = [(80, 0), (560, 480)]
        mid_height = 240
        with open(os.path.join(save_dir, "court_annotations.txt"), "w") as f:
            f.write(f"corners={corners}\n")
            f.write(f"roi_corners={roi_corners}\n")
            f.write(f"mid_height={mid_height}\n")

        fake = FakeFrameSource(frames=frames)
        adapter = StreamAdapter(source=fake, fps=30.0, first_frame=first)

        system = BadmintonAnalysisSystem(
            video_path="_e2e_test.mp4",
            template_path=template_path,
            output_dir=save_dir,
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
        )
        system.keep_audio = False

        try:
            system.process_video()
        finally:
            adapter.release()

        # Check that an output mp4 was produced
        out_mp4 = os.path.join(save_dir, "detect__e2e_test.mp4")
        self.assertTrue(os.path.exists(out_mp4), f"output mp4 missing: {out_mp4}")
        size = os.path.getsize(out_mp4)
        self.assertGreater(size, 1000, f"output mp4 too small: {size} bytes")
        # Check jsonl
        jsonl = os.path.join(save_dir, "detections.jsonl")
        self.assertTrue(os.path.exists(jsonl), f"jsonl missing: {jsonl}")


if __name__ == "__main__":
    unittest.main()
