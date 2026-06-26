"""Unit tests for StreamAdapter."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from badminton_analysis.sources.stream_adapter import StreamAdapter
from tests.fake_source import FakeFrameSource


class StreamAdapterTest(unittest.TestCase):
    def test_is_opened_initially_true(self):
        src = FakeFrameSource()
        a = StreamAdapter(src, fps=30.0)
        self.assertTrue(a.isOpened())

    def test_is_opened_false_after_release(self):
        src = FakeFrameSource()
        a = StreamAdapter(src, fps=30.0)
        a.release()
        self.assertFalse(a.isOpened())

    def test_release_calls_source_close(self):
        src = FakeFrameSource()
        a = StreamAdapter(src, fps=30.0)
        a.release()
        self.assertTrue(src.closed)

    def test_read_first_frame_prebuffered(self):
        first = FakeFrameSource.make_frame(720, 1280, 7)
        src = FakeFrameSource(frames=[FakeFrameSource.make_frame(100, 100, 1)])
        a = StreamAdapter(src, fps=30.0, first_frame=first)

        ok, frame = a.read()
        self.assertTrue(ok)
        self.assertIs(frame, first)

    def test_read_falls_through_to_source_after_first_frame(self):
        first = FakeFrameSource.make_frame(720, 1280)
        second = FakeFrameSource.make_frame(720, 1280, 42)
        src = FakeFrameSource(frames=[second])
        a = StreamAdapter(src, fps=30.0, first_frame=first)

        _, f1 = a.read()
        ok2, f2 = a.read()
        self.assertIs(f1, first)
        self.assertTrue(ok2)
        self.assertIs(f2, second)

    def test_read_returns_false_when_source_fails(self):
        src = FakeFrameSource(frames=[FakeFrameSource.make_frame(10, 10)], fail_after=0)
        a = StreamAdapter(src, fps=30.0)
        ok, frame = a.read()
        self.assertFalse(ok)
        self.assertIsNone(frame)

    def test_get_fps_returns_configured(self):
        src = FakeFrameSource()
        a = StreamAdapter(src, fps=25.0)
        self.assertEqual(a.get(cv2.CAP_PROP_FPS), 25.0)

    def test_get_width_height_uses_prebuffered_first_frame(self):
        first = FakeFrameSource.make_frame(480, 640)
        src = FakeFrameSource()
        a = StreamAdapter(src, fps=30.0, first_frame=first)
        self.assertEqual(a.get(cv2.CAP_PROP_FRAME_WIDTH), 640.0)
        self.assertEqual(a.get(cv2.CAP_PROP_FRAME_HEIGHT), 480.0)

    def test_get_frame_count_returns_minus_one(self):
        src = FakeFrameSource()
        a = StreamAdapter(src, fps=30.0)
        self.assertEqual(a.get(cv2.CAP_PROP_FRAME_COUNT), -1.0)

    def test_get_without_first_frame_returns_zero(self):
        src = FakeFrameSource(frames=[FakeFrameSource.make_frame(10, 20)])
        a = StreamAdapter(src, fps=30.0)
        self.assertEqual(a.get(cv2.CAP_PROP_FRAME_WIDTH), 0.0)
        self.assertEqual(a.get(cv2.CAP_PROP_FRAME_HEIGHT), 0.0)
        ok, _ = a.read()
        self.assertTrue(ok)
