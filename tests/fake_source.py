"""Test doubles for badminton_analysis.sources."""
from __future__ import annotations

import numpy as np

from badminton_analysis.sources.base import FrameResult, FrameSource


class FakeFrameSource(FrameSource):
    """A FrameSource that returns a scripted sequence of frames for testing."""

    source_type = "fake"

    def __init__(self, frames: list[np.ndarray] | None = None, fail_after: int | None = None):
        self._frames = list(frames or [])
        self._idx = 0
        self._fail_after = fail_after
        self.opened = False
        self.closed = False

    def open(self) -> bool:
        self.opened = True
        return True

    def next_frame(self) -> FrameResult:
        if self._fail_after is not None and self._idx >= self._fail_after:
            return FrameResult.failure("scripted failure", self.source_type, self._idx)
        if self._idx >= len(self._frames):
            return FrameResult.failure("end of script", self.source_type, self._idx)
        frame = self._frames[self._idx]
        self._idx += 1
        return FrameResult.success(frame, self._idx - 1, self.source_type)

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def make_frame(h: int, w: int, value: int = 0) -> np.ndarray:
        return np.full((h, w, 3), value, dtype=np.uint8)
