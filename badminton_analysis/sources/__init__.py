"""Real-time video sources for Good-Badminton."""
from .base import FrameResult, FrameSource
from .headless_browser import HeadlessBrowserSource
from .screen_capture import ScreenCaptureSource
from .stream_adapter import StreamAdapter
from .video_file import VideoFileSource

__all__ = [
    "FrameResult",
    "FrameSource",
    "HeadlessBrowserSource",
    "ScreenCaptureSource",
    "StreamAdapter",
    "VideoFileSource",
]
