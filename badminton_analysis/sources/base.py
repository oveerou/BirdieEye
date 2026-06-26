from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class FrameResult:
    ok: bool
    frame: np.ndarray | None
    frame_idx: int
    timestamp: float
    source_type: str
    error: str | None = None

    @classmethod
    def success(
        cls, frame: np.ndarray, frame_idx: int, source_type: str, timestamp: float | None = None
    ) -> "FrameResult":
        return cls(
            ok=True,
            frame=frame,
            frame_idx=frame_idx,
            timestamp=timestamp if timestamp is not None else time.time(),
            source_type=source_type,
            error=None,
        )

    @classmethod
    def failure(cls, error: str, source_type: str, frame_idx: int = -1) -> "FrameResult":
        return cls(
            ok=False,
            frame=None,
            frame_idx=frame_idx,
            timestamp=time.time(),
            source_type=source_type,
            error=error,
        )


class FrameSource(ABC):
    source_type: str = "base"

    @abstractmethod
    def open(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def next_frame(self) -> FrameResult:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __iter__(self) -> Iterator[FrameResult]:
        if not self.open():
            yield FrameResult.failure(f"无法打开视频源: {self.source_type}", self.source_type)
            return
        idx = 0
        while True:
            res = self.next_frame()
            if not res.ok:
                break
            res.frame_idx = idx
            yield res
            idx += 1
        self.close()

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
