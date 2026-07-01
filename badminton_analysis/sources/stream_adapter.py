"""Wraps a FrameSource to look like cv2.VideoCapture for BadmintonAnalysisSystem."""
from __future__ import annotations

import time

import cv2
import numpy as np

from .base import FrameSource


class StreamAdapter:
    """Adapter exposing the cv2.VideoCapture subset that process_video() uses.

    Methods provided:
        isOpened() -> bool
        read() -> (ok: bool, frame: np.ndarray | None)
        get(prop: int) -> float
        release() -> None

    The pre-buffered `first_frame` is served on the first `read()` call (so the
    consumer can size its video writer), and is also used by `get()` to answer
    CAP_PROP_FRAME_WIDTH / CAP_PROP_FRAME_HEIGHT without consuming a real frame.
    """

    def __init__(
        self,
        source: FrameSource,
        fps: float = 30.0,
        first_frame: np.ndarray | None = None,
    ):
        self._source = source
        self._fps = float(fps)
        self._first_frame = first_frame
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._first_frame is not None:
            frame = self._first_frame
            self._first_frame = None
            return True, frame
        # Retry up to 3 times to survive transient capture failures
        # (e.g., screen capture hiccup, window occlusion during angle switch)
        max_retries = 3
        for attempt in range(max_retries):
            res = self._source.next_frame()
            if res.ok:
                return True, res.frame
            # Brief pause before retry to let the source recover
            time.sleep(0.05)
        return False, None

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            if self._first_frame is None:
                return 0.0
            h, w = self._first_frame.shape[:2]
            return float(w if prop == cv2.CAP_PROP_FRAME_WIDTH else h)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return -1.0
        return 0.0

    def release(self) -> None:
        if self._opened:
            self._source.close()
            self._opened = False
