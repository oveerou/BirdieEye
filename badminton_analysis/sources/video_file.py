from __future__ import annotations

from pathlib import Path

import cv2

from .base import FrameResult, FrameSource


class VideoFileSource(FrameSource):
    source_type = "video_file"

    def __init__(self, video_path: str | Path):
        self.video_path = str(video_path)
        self._cap: cv2.VideoCapture | None = None
        self._frame_idx = 0

    def open(self) -> bool:
        path = Path(self.video_path)
        if not path.exists():
            return False
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._frame_idx = 0
        return True

    def next_frame(self) -> FrameResult:
        if self._cap is None:
            return FrameResult.failure("视频未打开", self.source_type)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return FrameResult.failure("视频已结束或读取失败", self.source_type, self._frame_idx)
        res = FrameResult.success(frame, self._frame_idx, self.source_type)
        self._frame_idx += 1
        return res

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)

    @property
    def frame_count(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
