from __future__ import annotations

import numpy as np

from .base import FrameResult, FrameSource


class ScreenCaptureSource(FrameSource):
    source_type = "screen_capture"

    def __init__(self, left: int = 100, top: int = 100, width: int = 1280, height: int = 720):
        self.region = {"left": left, "top": top, "width": width, "height": height}
        self._sct = None
        self._frame_idx = 0

    def open(self) -> bool:
        if self.region["width"] <= 0 or self.region["height"] <= 0:
            return False
        try:
            import mss

            self._sct = mss.MSS()
        except Exception:
            self._sct = None
            return False
        self._frame_idx = 0
        return True

    def next_frame(self) -> FrameResult:
        if self._sct is None:
            return FrameResult.failure("屏幕捕获未打开", self.source_type)
        try:
            shot = self._sct.grab(self.region)
        except Exception as e:
            return FrameResult.failure(f"屏幕捕获失败: {e}", self.source_type, self._frame_idx)
        img = np.frombuffer(shot.rgb, dtype=np.uint8)
        if img.size == 0:
            return FrameResult.failure("屏幕捕获空帧", self.source_type, self._frame_idx)
        h, w = self.region["height"], self.region["width"]
        frame = img.reshape(h, w, 3)
        frame = frame[:, :, ::-1].copy()
        res = FrameResult.success(frame, self._frame_idx, self.source_type)
        self._frame_idx += 1
        return res

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
