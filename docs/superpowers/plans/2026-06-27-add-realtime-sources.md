# Add Real-Time Sources to Good-Badminton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable `mss` screen capture and headless-browser sources as additional inputs to Good-Badminton, without breaking the existing `python main.py --video-path xxx.mp4` workflow.

**Architecture:** Copy the `FrameSource` abstraction from `D:\A_shixi\football-realtime-analyzer\football_analyzer\sources\` into a new `badminton_analysis/sources/` package. Add a `StreamAdapter` that wraps a `FrameSource` to expose the `cv2.VideoCapture` interface (`isOpened / read / get / release`) used by `BadmintonAnalysisSystem.process_video()`. Add one optional `frame_source` parameter to `BadmintonAnalysisSystem` (default `None` preserves existing behavior). Add a new entry point `python -m badminton_analysis.live` that wires these together.

**Tech Stack:** Python 3.12, OpenCV 4.13, mss â‰¥10, websocket-client, requests, stdlib `unittest` for tests, Git for version control.

**Spec:** `docs/superpowers/specs/2026-06-27-add-realtime-sources-design.md`

## Global Constraints

- Existing reference: `python main.py --video-path videos/demo1.mp4 --template-path templates/demo1.png --display false` produces `outputs/demo1/detect_demo1.mp4` of exactly **22,112,015 bytes** (regression test baseline).
- `BadmintonAnalysisSystem.__init__` adds one new keyword arg `frame_source=None`; no other signature changes.
- `process_video()` adds at most 2 lines: one cap-source branch and one `total_frames > 0` guard.
- New entry point is `python -m badminton_analysis.live` â€” `main.py` is **not** modified.
- New dependencies pinned in `requirements.txt`: `mss>=10.0.0`, `websocket-client`, `requests>=2.28`.
- Live source output uses `keep_audio=False` (existing `_cleanup` branch handles this).
- All commits are local; do not push.
- Working directory for all commands: `D:\A_shixi\Good-Badminton` (PowerShell).
- venv Python: `D:\A_shixi\Good-Badminton\.venv\Scripts\python.exe`.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `badminton_analysis/sources/__init__.py` | Create | Package facade; re-exports public API |
| `badminton_analysis/sources/base.py` | Create | `FrameSource` ABC + `FrameResult` dataclass (verbatim from football) |
| `badminton_analysis/sources/screen_capture.py` | Create | `ScreenCaptureSource` (verbatim from football) |
| `badminton_analysis/sources/headless_browser.py` | Create | `HeadlessBrowserSource` (verbatim from football) |
| `badminton_analysis/sources/stream_adapter.py` | Create | `StreamAdapter` (TDD; cv2.VideoCapture-shaped) |
| `badminton_analysis/live.py` | Create | `python -m badminton_analysis.live` entry point |
| `badminton_analysis/system.py` | Modify | +1 init arg, +1 cap branch, +1 guard |
| `requirements.txt` | Modify | +3 deps |
| `README.md` | Modify | +"Real-time sources" section |
| `tests/__init__.py` | Create | Empty; makes tests a package |
| `tests/test_stream_adapter.py` | Create | Unit tests for `StreamAdapter` (TDD) |
| `tests/fake_source.py` | Create | `FakeFrameSource` test double |

---

### Task 1: Add new dependencies to requirements.txt

**Files:**
- Modify: `requirements.txt` (append three lines)

**Why first:** The remaining tasks `import mss`, `import websocket`, and `import requests`; these need to be installable before any import-time check.

- [ ] **Step 1: Append three lines to requirements.txt**

Append (do not rewrite the file) these three lines at the end of `requirements.txt`:

```
mss>=10.0.0
websocket-client
requests>=2.28
```

Final file content (use `Get-Content requirements.txt` to verify):

```
opencv-python==4.10.0.84
opencv-contrib-python==4.10.0.84
numpy>=1.21.6,<2.0
pillow>=9.2.0,<12.0
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.5.1+cu121
torchvision==0.20.1+cu121
ultralytics>=8.0.0
rtmlib>=0.0.1
onnxruntime-gpu==1.20.1
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
moviepy>=1.0.3,<2.0
scipy>=1.7.0
scikit-learn>=1.0.0
openpyxl>=3.0.0
mss>=10.0.0
websocket-client
requests>=2.28
```

- [ ] **Step 2: Install the new packages into the venv**

Run:
```powershell
.\.venv\Scripts\python.exe -m pip install mss>=10.0.0 websocket-client requests>=2.28
```

Expected: `Successfully installed mss-X.Y.Z websocket-client-X.Y.Z requests-X.Y.Z` (versions vary).

- [ ] **Step 3: Verify imports work**

Run:
```powershell
.\.venv\Scripts\python.exe -c "import mss, websocket, requests; print('mss:', mss.__version__); print('websocket-client:', websocket.__version__); print('requests:', requests.__version__)"
```

Expected: Three version lines, no errors.

- [ ] **Step 4: Commit**

```powershell
git add requirements.txt
git commit -m "deps: add mss, websocket-client, requests for real-time sources"
```

---

### Task 2: Create sources/ package skeleton with base.py

**Files:**
- Create: `badminton_analysis/sources/__init__.py` (empty)
- Create: `badminton_analysis/sources/base.py` (copy from football)

**Why before other sources:** The `FrameSource` ABC and `FrameResult` dataclass are the contract every other source in the package implements. Copying `base.py` first makes the package importable and gives later tasks a concrete type to inherit from.

- [ ] **Step 1: Read the source file we are copying**

Read `D:\A_shixi\football-realtime-analyzer\football_analyzer\sources\base.py` and verify it contains `FrameResult` (a `@dataclass`) and `FrameSource` (an `ABC` with `open`, `next_frame`, `close` abstract methods plus `__iter__`, `__enter__`, `__exit__`).

- [ ] **Step 2: Create the empty package init file**

Create `badminton_analysis/sources/__init__.py` with a single docstring (no exports yet ¡ª wiring happens in Task 6):

```python
"""Real-time video sources for Good-Badminton."""
```

- [ ] **Step 3: Copy base.py verbatim from the football project**

Read `D:\A_shixi\football-realtime-analyzer\football_analyzer\sources\base.py` and write the **exact same content** to `badminton_analysis/sources/base.py`. Do not modify, refactor, or "improve" anything. The file must contain:

```python
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
            yield FrameResult.failure(f"ÎŞ·¨´ò¿ªÊÓÆµÔ´: {self.source_type}", self.source_type)
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
```

- [ ] **Step 4: Verify the package imports**

Run:
```powershell
.\.venv\Scripts\python.exe -c "from badminton_analysis.sources.base import FrameSource, FrameResult; print('FrameSource:', FrameSource); print('FrameResult:', FrameResult)"
```

Expected: Two lines, no `ImportError`.

- [ ] **Step 5: Commit**

```powershell
git add badminton_analysis/sources/__init__.py badminton_analysis/sources/base.py
git commit -m "feat(sources): add sources package with FrameSource/FrameResult"
```

---

### Task 3: Add ScreenCaptureSource (copy from football)

**Files:**
- Create: `badminton_analysis/sources/screen_capture.py` (verbatim copy)

**Why now:** Needed by Task 5 (StreamAdapter tests may use it indirectly) and Task 9 (live.py smoke test).

- [ ] **Step 1: Read the source**

Read `D:\A_shixi\football-realtime-analyzer\football_analyzer\sources\screen_capture.py` and confirm the class is `ScreenCaptureSource(source_type="screen_capture")` with `__init__(left, top, width, height)` and `open/next_frame/close` methods that use `mss.MSS().grab(...)` and `np.frombuffer`.

- [ ] **Step 2: Copy verbatim**

Write the **exact same content** (no edits) to `badminton_analysis/sources/screen_capture.py`. Use the `read` tool to copy the football version and the `write` tool to save it. The file should be ~52 lines.

The class must implement:
- `source_type = "screen_capture"`
- `__init__(self, left: int = 100, top: int = 100, width: int = 1280, height: int = 720)`
- `open()` returning `True` if `mss.MSS()` instantiated, `False` otherwise
- `next_frame()` returning `FrameResult.success(frame, idx, "screen_capture")` after a `sct.grab(self.region)`, or `FrameResult.failure(...)` on error
- `close()` calling `sct.close()` if `_sct is not None`

- [ ] **Step 3: Verify import and instantiation**

Run:
```powershell
.\.venv\Scripts\python.exe -c "from badminton_analysis.sources.screen_capture import ScreenCaptureSource; s = ScreenCaptureSource(); print('class:', type(s).__name__); print('source_type:', s.source_type)"
```

Expected:
```
class: ScreenCaptureSource
source_type: screen_capture
```

(No need to call `open()` here ¡ª it would grab the actual desktop, which is undesirable in this smoke test.)

- [ ] **Step 4: Commit**

```powershell
git add badminton_analysis/sources/screen_capture.py
git commit -m "feat(sources): add ScreenCaptureSource (mss)"
```

---

### Task 4: Add HeadlessBrowserSource (copy from football)

**Files:**
- Create: `badminton_analysis/sources/headless_browser.py` (verbatim copy)

**Why now:** Same as Task 3 ¡ª needed by Task 9 smoke test and completes the source set so Task 6 can wire up the package.

- [ ] **Step 1: Read the source**

Read `D:\A_shixi\football-realtime-analyzer\football_analyzer\sources\headless_browser.py` and confirm:
- Class `HeadlessBrowserSource(source_type="browser_headless")`
- `__init__(url, chrome_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe", port=0, width=1280, height=720, wait_sec=10.0)`
- `open()` spawns `chrome --headless=new --remote-debugging-port=...`, polls `http://127.0.0.1:<port>/json/list`, opens WebSocket via `websocket.create_connection`
- `next_frame()` calls `_grab_video_frame()` (CDP `Runtime.evaluate` of a JS snippet that reads `<video>` to canvas), falls back to `_grab_screenshot()` (CDP `Page.captureScreenshot`)
- `close()` closes the WebSocket and terminates the Chrome process

- [ ] **Step 2: Copy verbatim**

Write the **exact same content** (no edits) to `badminton_analysis/sources/headless_browser.py`. ~161 lines. Uses `subprocess`, `base64`, `json` (all stdlib), `time` (stdlib), `numpy`, `requests`, `cv2`, `websocket`.

- [ ] **Step 3: Verify import**

Run:
```powershell
.\.venv\Scripts\python.exe -c "from badminton_analysis.sources.headless_browser import HeadlessBrowserSource; s = HeadlessBrowserSource('https://example.com'); print('class:', type(s).__name__); print('source_type:', s.source_type); print('chrome_path:', s.chrome_path)"
```

Expected:
```
class: HeadlessBrowserSource
source_type: browser_headless
chrome_path: C:\Program Files\Google\Chrome\Application\chrome.exe
```

(Do **not** call `open()` here ¡ª it would spawn a real Chrome process.)

- [ ] **Step 4: Commit**

```powershell
git add badminton_analysis/sources/headless_browser.py
git commit -m "feat(sources): add HeadlessBrowserSource (Chrome CDP)"
```

---

### Task 5: TDD - StreamAdapter

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/fake_source.py`
- Create: `tests/test_stream_adapter.py`
- Create: `badminton_analysis/sources/stream_adapter.py`

**Why TDD here:** `StreamAdapter` is the bridge between the new `FrameSource` world and the existing `cv2.VideoCapture` consumer. Its contract is small and well-defined, so unit tests with a fake `FrameSource` give us a fast feedback loop without needing mss, Chrome, or a display. Subsequent tasks depend on this contract.

- [ ] **Step 1: Create the empty tests package**

Create `tests/__init__.py`:

```python
"""Tests for Good-Badminton."""
```

- [ ] **Step 2: Write the FakeFrameSource test double first**

Create `tests/fake_source.py`:

```python
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
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_stream_adapter.py`:

```python
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

        _, f1 = a.read()              # pre-buffered
        ok2, f2 = a.read()            # from source
        self.assertIs(f1, first)
        self.assertTrue(ok2)
        self.assertIs(f2, second)

    def test_read_returns_false_when_source_fails(self):
        src = FakeFrameSource(frames=[FakeFrameSource.make_frame(10, 10)], fail_after=0)
        a = StreamAdapter(src, fps=30.0)
        # first call: fail_after=0 ¡ú failure
        ok, frame = a.read()
        self.assertFalse(ok)
        self.assertIsNone(frame)

    def test_get_fps_returns_configured(self):
        src = FakeFrameSource()
        a = StreamAdapter(src, fps=25.0)
        self.assertEqual(a.get(cv2.CAP_PROP_FPS), 25.0)

    def test_get_width_height_uses_prebuffered_first_frame(self):
        first = FakeFrameSource.make_frame(480, 640)   # h=480, w=640
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
        # No pre-buffered frame; get() should NOT consume from the source.
        self.assertEqual(a.get(cv2.CAP_PROP_FRAME_WIDTH), 0.0)
        self.assertEqual(a.get(cv2.CAP_PROP_FRAME_HEIGHT), 0.0)
        # Source is untouched:
        ok, _ = a.read()
        self.assertTrue(ok)
```

- [ ] **Step 4: Run tests and confirm they fail (red)**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_stream_adapter -v
```

Expected: `ModuleNotFoundError: No module named 'badminton_analysis.sources.stream_adapter'` (or `ImportError` of the same kind). Every test should fail at import.

- [ ] **Step 5: Write the minimal StreamAdapter implementation**

Create `badminton_analysis/sources/stream_adapter.py`:

```python
"""Wraps a FrameSource to look like cv2.VideoCapture for BadmintonAnalysisSystem."""
from __future__ import annotations

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
        res = self._source.next_frame()
        if not res.ok:
            return False, None
        return True, res.frame

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
```

- [ ] **Step 6: Run tests and confirm they pass (green)**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_stream_adapter -v
```

Expected: 10 tests, all `ok`. If any fail, fix the implementation (not the tests) and re-run.

- [ ] **Step 7: Commit**

```powershell
git add badminton_analysis/sources/stream_adapter.py tests/__init__.py tests/fake_source.py tests/test_stream_adapter.py
git commit -m "feat(sources): add StreamAdapter with unit tests"
```

---

### Task 6: Wire the sources package public API

**Files:**
- Modify: `badminton_analysis/sources/__init__.py`

**Why now:** `system.py`, `live.py`, and any future import site use `from badminton_analysis.sources import ...` rather than reaching into submodules.

- [ ] **Step 1: Replace the placeholder with full re-exports**

Write `badminton_analysis/sources/__init__.py` as:

```python
"""Real-time video sources for Good-Badminton."""
from .base import FrameResult, FrameSource
from .headless_browser import HeadlessBrowserSource
from .screen_capture import ScreenCaptureSource
from .stream_adapter import StreamAdapter

__all__ = [
    "FrameResult",
    "FrameSource",
    "HeadlessBrowserSource",
    "ScreenCaptureSource",
    "StreamAdapter",
]
```

- [ ] **Step 2: Verify all public symbols import via the package**

Run:
```powershell
.\.venv\Scripts\python.exe -c "from badminton_analysis.sources import FrameSource, FrameResult, ScreenCaptureSource, HeadlessBrowserSource, StreamAdapter; print('all imports OK')"
```

Expected: `all imports OK`.

- [ ] **Step 3: Re-run the StreamAdapter unit tests**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_stream_adapter -v
```

Expected: 10 tests, all `ok` (no regressions).

- [ ] **Step 4: Commit**

```powershell
git add badminton_analysis/sources/__init__.py
git commit -m "feat(sources): wire package public API"
```

---

### Task 7: Modify BadmintonAnalysisSystem for the new frame_source path

**Files:**
- Modify: `badminton_analysis/system.py:62-99` (__init__ signature + new attribute)
- Modify: `badminton_analysis/system.py:162-174` (process_video cap branch + duration guard)

**Why this exact diff:** Two surgical edits preserve the entire existing analysis pipeline. `frame_source=None` keeps `main.py` callers working without change.

- [ ] **Step 1: Read the current __init__ signature (lines 62-99) and process_video head (lines 162-174)**

Use `read` with the appropriate offset/limit. Confirm:
- The `__init__` ends with `..., show_pose_roi=True):` on line 69.
- The `process_video` opens with `self.start_time = time.time()` then `cap = cv2.VideoCapture(self.video_path)`.

- [ ] **Step 2: Add `frame_source=None` to the __init__ signature**

Edit line 69 of `badminton_analysis/system.py` (the end of the parameter list). Change:

```python
                 yolo_pose_model='yolo11n-pose.pt', show_pose_roi=True):
```

to:

```python
                 yolo_pose_model='yolo11n-pose.pt', show_pose_roi=True,
                 frame_source=None):
```

- [ ] **Step 3: Store the new attribute near the other `self.*` assignments**

After the existing `self.show_pose_roi = show_pose_roi` line (line 78), add:

```python
        self.frame_source = frame_source
```

- [ ] **Step 4: Replace the `cap =` line in process_video**

Find line 166 of `badminton_analysis/system.py`:

```python
        cap = cv2.VideoCapture(self.video_path)
```

Replace with:

```python
        cap = self.frame_source if self.frame_source is not None else cv2.VideoCapture(self.video_path)
```

- [ ] **Step 5: Add the `total_frames > 0` guard for `video_duration`**

Find lines 171-174 of `badminton_analysis/system.py`:

```python
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            raise RuntimeError(f"Unable to read FPS from video: {self.video_path}")
        video_duration = total_frames / fps
```

Replace with:

```python
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            raise RuntimeError(f"Unable to read FPS from video: {self.video_path}")
        if total_frames > 0:
            video_duration = total_frames / fps
        else:
            video_duration = 0
```

- [ ] **Step 6: Verify the module still imports**

Run:
```powershell
.\.venv\Scripts\python.exe -c "from badminton_analysis.system import BadmintonAnalysisSystem; import inspect; sig = inspect.signature(BadmintonAnalysisSystem.__init__); print('frame_source' in sig.parameters); print('last 3 params:', list(sig.parameters.keys())[-3:])"
```

Expected:
```
True
['yolo_pose_model', 'show_pose_roi', 'frame_source']
```

- [ ] **Step 7: Commit**

```powershell
git add badminton_analysis/system.py
git commit -m "feat(system): accept optional frame_source for live sources"
```

---

### Task 8: Regression test - existing local file flow must be byte-identical

**Files:** none modified (this task is verification only)

**Why this gate:** The system.py diff in Task 7 must not change the output for the existing CLI. This is the strongest possible behavioral test.

- [ ] **Step 1: Remove any prior outputs to force regeneration**

Run:
```powershell
Remove-Item -LiteralPath outputs/demo1 -Recurse -Force -ErrorAction SilentlyContinue
```

- [ ] **Step 2: Run the standard local-file command**

Run:
```powershell
.\.venv\Scripts\python.exe main.py --video-path videos/demo1.mp4 --template-path templates/demo1.png --display false
```

Expected: completes in ~50s on GPU, prints `´¦ÀíºÄÊ±: ~52 Ãë` and `Video saved to: outputs\demo1\detect_demo1.mp4`.

- [ ] **Step 3: Verify byte sizes match the historical reference**

Run:
```powershell
(Get-Item outputs/demo1/detect_demo1.mp4).Length
(Get-Item outputs/demo1/detections.jsonl).Length
(Get-Item outputs/demo1/metadata.json).Length
```

Expected:
```
22112015
467583
944
```

(All three numbers must match exactly. If any differs, the system.py change broke something ¡ª re-check Steps 2-5 of Task 7 before proceeding.)

- [ ] **Step 4: Re-run unit tests to confirm no regressions in the test suite**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests -v
```

Expected: 10 tests, all `ok`.

- [ ] **Step 5: No commit (this is verification only)**

If all checks pass, proceed to Task 9. If any check fails, the implementation is wrong ¡ª do not advance.

---

### Task 9: Create live.py entry point

**Files:**
- Create: `badminton_analysis/live.py`

**Why now:** This is the script users will actually run. It composes the source, adapter, and `BadmintonAnalysisSystem`, handles first-frame capture and Ctrl+C gracefully.

- [ ] **Step 1: Write live.py**

Create `badminton_analysis/live.py` with the following content (verbatim):

```python
"""Real-time source entry point for Good-Badminton.

Usage:
    python -m badminton_analysis.live --source screen_capture \\
           --region 100,100,1280,720 --fps 30 --display true

    python -m badminton_analysis.live --source browser_headless \\
           --url https://example.com/live --fps 30 --display true
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import cv2

from .sources import HeadlessBrowserSource, ScreenCaptureSource, StreamAdapter


def build_source(args: argparse.Namespace):
    if args.source == "screen_capture":
        left, top, width, height = args.region
        return ScreenCaptureSource(left=left, top=top, width=width, height=height)
    if args.source == "browser_headless":
        return HeadlessBrowserSource(
            url=args.url,
            chrome_path=args.chrome_path,
            width=args.browser_w,
            height=args.browser_h,
            wait_sec=args.wait_sec,
        )
    raise ValueError(f"Unknown source: {args.source}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Good-Badminton real-time source")
    p.add_argument("--source", required=True, choices=["screen_capture", "browser_headless"])
    p.add_argument("--fps", type=float, default=30.0, help="Output video fps (default 30)")
    p.add_argument("--display", choices=["true", "false"], default="true")
    p.add_argument("--language", default="zh", choices=["zh", "en"])
    p.add_argument("--pose-family", default="yolo-pose", choices=["rtmpose", "rtmo", "yolo-pose"])
    p.add_argument("--pose-mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    p.add_argument("--yolo-pose-model", default="weights/yolo11n-pose.pt")
    p.add_argument("--ball-model", default="weights/yolo11s-ball.pt")
    p.add_argument("--display-overlay", choices=["true", "false"], default="true",
                   help="Whether to draw overlays (skeletons/trajectories)")
    p.add_argument("--region", type=int, nargs=4, default=[100, 100, 1280, 720],
                   metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
                   help="(screen_capture) region to grab")
    p.add_argument("--url", default="", help="(browser_headless) URL to open")
    p.add_argument("--chrome-path", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                   help="(browser_headless) chrome.exe path")
    p.add_argument("--browser-w", type=int, default=1280)
    p.add_argument("--browser-h", type=int, default=720)
    p.add_argument("--wait-sec", type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from .system import BadmintonAnalysisSystem, load_runtime_dependencies
    load_runtime_dependencies()

    run_id = f"live_{args.source}_{time.strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join("outputs", run_id)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[live] run_id={run_id}")
    print(f"[live] opening source: {args.source}")
    source = build_source(args)
    if not source.open():
        raise SystemExit(f"[live] failed to open {args.source}")
    print("[live] source opened")

    print("[live] grabbing first frame (used as court template source)...")
    first = source.next_frame()
    if not first.ok or first.frame is None:
        source.close()
        raise SystemExit(f"[live] failed to grab first frame: {first.error}")

    first_frame_path = os.path.join(save_dir, "first_frame.png")
    cv2.imwrite(first_frame_path, first.frame)
    print(f"[live] first frame saved: {first_frame_path} "
          f"({first.frame.shape[1]}x{first.frame.shape[0]})")

    adapter = StreamAdapter(source=source, fps=args.fps, first_frame=first.frame)

    system = BadmintonAnalysisSystem(
        video_path=f"{run_id}.mp4",
        template_path=first_frame_path,
        output_dir=save_dir,
        ball_model_path=args.ball_model,
        pose_family=args.pose_family,
        pose_mode=args.pose_mode,
        yolo_pose_model=args.yolo_pose_model,
        language=args.language,
        show_display=args.display == "true",
        show_skeletons=args.display_overlay == "true",
        show_player_trajectories=args.display_overlay == "true",
        show_court_trajectory=args.display_overlay == "true",
        show_shuttlecock_trajectory=args.display_overlay == "true",
        show_player_stats=args.display_overlay == "true",
        frame_source=adapter,
    )
    system.keep_audio = False

    try:
        system.process_video()
        print(f"[live] finished: {system.output_video_path}")
    except KeyboardInterrupt:
        print("\n[live] interrupted by user (Ctrl+C)")
        try:
            if hasattr(system, "video_writer") and system.video_writer is not None:
                system.video_writer.release()
        except Exception:
            pass
        if os.path.exists(system.temp_output_video_path):
            try:
                shutil.copy2(system.temp_output_video_path, system.output_video_path)
                print(f"[live] partial output saved: {system.output_video_path}")
            except Exception as e:
                print(f"[live] failed to copy partial output: {e}")
    finally:
        adapter.release()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify --help works (and fast-imports are avoided)**

Run:
```powershell
.\.venv\Scripts\python.exe -m badminton_analysis.live --help
```

Expected: prints the full argparser help, no traceback, exits 0.

- [ ] **Step 3: Verify an invalid source is rejected**

Run:
```powershell
.\.venv\Scripts\python.exe -m badminton_analysis.live --source bogus
```

Expected: exits non-zero with `argument --source: invalid choice: 'bogus'`.

- [ ] **Step 4: Commit**

```powershell
git add badminton_analysis/live.py
git commit -m "feat(live): add real-time source entry point"
```

---

### Task 10: mss source smoke test (manual)

**Files:** none (manual verification on the user's desktop)

**Why a manual task:** A real screen capture needs a visible window on the user's display; we cannot simulate that from CI / headless. The plan author should perform this on the same Windows machine that has the venv.

- [ ] **Step 1: Stage a known visible window**

Open any visible window with stable content (e.g., Notepad with some text, or a paused video player) and position it so it overlaps the region `100,100 ¡ú 1280,720`. Confirm you can see the window in that area.

- [ ] **Step 2: Run the live source for ~5 seconds**

In one PowerShell window run:

```powershell
.\.venv\Scripts\python.exe -m badminton_analysis.live --source screen_capture --region 100,100,1280,720 --fps 30 --display true
```

Expected within 1-2 seconds:
- Console prints `[live] source opened` and `[live] first frame saved: ...`
- An OpenCV window opens titled `frame` showing the captured screen region
- After ~2-5s the auto court-detect prompt (or a "click 4 corners" OpenCV window) appears

- [ ] **Step 3: Complete the court annotation**

If the auto court preview shows a green box, press `Enter` (or `Y`). If not, press `M` (or `R` or `Esc`) and click the 4 court corners in the original-image coordinate space.

- [ ] **Step 4: Let the analysis run for a few more seconds, then Ctrl+C**

Press `Ctrl+C` in the PowerShell window.

Expected:
- Console prints `[live] interrupted by user (Ctrl+C)` then `[live] partial output saved: outputs/live_screen_capture_<timestamp>/detect_*.mp4`
- The `detect_*.mp4` file exists with size > 1 MB
- The OpenCV window closes

- [ ] **Step 5: Verify the output file is playable**

Run:
```powershell
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 outputs/live_screen_capture_*/detect_*.mp4
```

Expected: prints `size=<bytes>` and `duration=<seconds>` (duration will be < 1s because Ctrl+C cuts it short, but the file should still be readable).

- [ ] **Step 6: No commit (manual verification)**

If any step fails, the bug is most likely in `stream_adapter.py` (re-check Task 5 unit tests) or in `system.py`'s `process_video` interaction with the adapter (re-check Task 7). If the source fails to open at all (`mss` not installed, region out of bounds), re-run Task 1's `pip install mss`.

---

### Task 11: Browser source smoke test (manual, optional)

**Files:** none (manual verification on the user's desktop)

**Why optional / why last:** This requires Chrome to be installed at the default path (or `--chrome-path` overridden) AND a reachable URL with a `<video>` element. Both are environment-dependent. If Chrome is not available, skip this task ¡ª the unit tests in Task 5 already cover the source plumbing, and the football project exercises the same `HeadlessBrowserSource` end-to-end.

- [ ] **Step 1: Confirm Chrome is installed**

Run:
```powershell
Test-Path "C:\Program Files\Google\Chrome\Application\chrome.exe"
```

If `False`, try `C:\Program Files (x86)\Google\Chrome\Application\chrome.exe` and pass the result via `--chrome-path`. If neither exists, **skip this entire task** (it is marked optional).

- [ ] **Step 2: Run the browser source against a test video URL**

In one PowerShell window run:

```powershell
.\.venv\Scripts\python.exe -m badminton_analysis.live --source browser_headless `
    --url "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4" `
    --fps 30 --display true
```

Expected within ~5-10 seconds:
- Console prints `[live] source opened` (this means Chrome started and the WebSocket handshake succeeded)
- `[live] first frame saved: ...` (a real frame from Big Buck Bunny was extracted)
- An OpenCV window shows the Big Buck Bunny frames with overlays

- [ ] **Step 3: Complete the court annotation**

Same as Task 10 Step 3.

- [ ] **Step 4: Run for ~8 seconds, then Ctrl+C**

Press `Ctrl+C` in the PowerShell window.

Expected:
- `[live] partial output saved: outputs/live_browser_headless_<timestamp>/detect_*.mp4`
- File exists, size > 1 MB
- The output mp4 should show actual Big Buck Bunny frames with analysis overlays

- [ ] **Step 5: If you get "failed to open" or "failed to grab first frame"**

Most common causes:
- Chrome path wrong ¡ú re-check `--chrome-path`
- URL not reachable ¡ú try the Big Buck Bunny URL above
- CORS blocks `<video>` capture ¡ú the source falls back to full-page screenshot (less useful but should not error)
- `wait_sec` too short ¡ú set `--wait-sec 20`

- [ ] **Step 6: No commit (manual verification)**

---

### Task 12: Document the new feature in README.md

**Files:**
- Modify: `README.md` (append a "Real-time sources" section near the end, before "Acknowledgments" or "License")

**Why last:** Documentation reflects what was actually built and verified, not what was planned. Writing it after Tasks 8-11 ensures the docs match reality.

- [ ] **Step 1: Find the right insertion point in README.md**

Run:
```powershell
Select-String -Path README.md -Pattern "^## " | ForEach-Object { $_.Line }
```

Identify the last `##` section before the "License" or "Acknowledgments" section. Note its line number. The new section will be appended just before it.

- [ ] **Step 2: Append the Real-time sources section**

Append the following to `README.md` (locate the right insertion point from Step 1 and insert before the last "Acknowledgments" / "License" section):

```markdown
## å®æ—¶æº (Real-time sources)

é™¤äº†æœ¬åœ°è§†é¢‘æ–‡ä»¶ï¼Œä¹Ÿå¯ä»¥ç›´æ¥ä»å±å¹•åŒºåŸŸæˆ–ç½‘é¡µç›´æ’­è·å–ç”»é¢ã€‚ä¸¤è€…éƒ½é€šè¿‡æ–°å…¥å£ `python -m badminton_analysis.live` å¯åŠ¨ï¼Œ**ä¸ä¼šæ”¹åŠ¨**ç°æœ‰çš„ `python main.py --video-path xxx.mp4` æµç¨‹ã€‚

æ–°å¢çš„ä¾èµ–ï¼š`mss`ï¼ˆå±å¹•æ•è·ï¼‰ã€`websocket-client` + `requests`ï¼ˆæ— å¤´æµè§ˆå™¨ï¼‰ã€‚å·²åœ¨ `requirements.txt` ä¸­å›ºå®šã€‚

### å±å¹•æ•è· (mss)

æŠ“å–å±å¹•æŒ‡å®šåŒºåŸŸï¼Œå¯ç”¨äºåˆ†ææœ¬åœ°è§†é¢‘æ’­æ”¾å™¨çª—å£ã€è®­ç»ƒè½¯ä»¶ç­‰ã€‚

```bat
python -m badminton_analysis.live --source screen_capture --region 100,100,1280,720 --fps 30 --display true
```

²ÎÊı `--region left top width height` Ö¸¶¨×¥È¡¾ØĞÎ£¬Ä¬ÈÏ `100 100 1280 720`¡£ÔËĞĞÊ±»á×Ô¶¯×¥Ê×Ö¡×÷ÎªÇò³¡±ê×¢Ô´£¨×Ô¶¯¼ì²â»òÊÖ¶¯µã 4 ½Çµã£©¡£

### ÍøÒ³Ö±²¥ (ÎŞÍ·ä¯ÀÀÆ÷)

ÓÃ headless Chrome ´ò¿ªº¬ `<video>` ÔªËØµÄÍøÒ³£¬Í¨¹ı Chrome DevTools Protocol ×¥È¡ video Ö¡¡£**²»ÒÀÀµ selenium**£¬Ö»ÓÃ `subprocess + websocket-client`¡£

```bat
python -m badminton_analysis.live --source browser_headless --url "https://example.com/live" --fps 30 --display true
```

Ç°ÖÃ£ºÏµÍ³ÒÑ°²×° Google Chrome£¨Ä¬ÈÏÂ·¾¶ `C:\Program Files\Google\Chrome\Application\chrome.exe`£¬¿ÉÍ¨¹ı `--chrome-path` ¸²¸Ç£©¡£ÆäËû²ÎÊı£º`--wait-sec 20` Ôö´óµÈ´ıÊ±¼ä¡¢`--browser-w/--browser-h` ÉèÖÃ´°¿Ú´óĞ¡¡£

### Í£Ö¹ÓëÊä³ö

°´ `Ctrl+C` ¼´¿ÉÓÅÑÅÍ£Ö¹¡£½áÊøºóÊä³öÔÚ `outputs/live_<source>_<timestamp>/detect_*.mp4`£¨ÎŞÒôÆµ£»Í¬ `keep_audio=false` Â·¾¶£©¡£

### ÒÑÖªÏŞÖÆ

- Êä³ö mp4 µÄ fps ¹Ì¶¨Îª `--fps` ²ÎÊı£¨Ä¬ÈÏ 30£©£¬ÓëÊµ¼ÊÔ´Ö¡ÂÊ²»Ò»ÖÂÊ±²¥·ÅËÙ¶È»áÓĞÆ«²î
- Ê×Ö¡ÊÇÇò³¡±ê×¢µÄÀ´Ô´£¬»á³öÏÖÔÚÊä³öÊÓÆµ¿ªÍ·
- Chrome Â·¾¶·ÇÄ¬ÈÏÊ±ĞèÓÃ `--chrome-path` ÏÔÊ½Ö¸¶¨
- macOS / Linux ÆÁÄ»²¶»ñÎ´¾­²âÊÔ£¨mss ¿âÖ§³Öµ«Ğè×ÔĞĞÑéÖ¤£©
```

- [ ] **Step 3: Verify the section was added in the right place**

Run:
```powershell
Select-String -Path README.md -Pattern "## ÊµÊ±Ô´"
```

Expected: prints the line number where the new section starts.

- [ ] **Step 4: Mirror the change to README_EN.md if it exists**

Run:
```powershell
Test-Path README_EN.md
```

If `True`, repeat Step 2 with an English version of the section. The English text should cover the same points:

```markdown
## Real-time sources

In addition to local video files, the analysis can also run on a screen region or a webpage with a `<video>` element. Both use a new entry point `python -m badminton_analysis.live` and do **not** change the existing `python main.py --video-path xxx.mp4` workflow.

New dependencies: `mss` (screen capture), `websocket-client` + `requests` (headless browser). Pinned in `requirements.txt`.

### Screen capture (mss)

```bat
python -m badminton_analysis.live --source screen_capture --region 100,100,1280,720 --fps 30 --display true
```

`--region left top width height` selects the rectangle to grab (default `100 100 1280 720`). The first frame is auto-captured and used as the court template source (auto-detect or manual 4-corner click).

### Web page live stream (headless browser)

Uses headless Chrome with the Chrome DevTools Protocol to pull `<video>` frames. **No selenium dependency**; only `subprocess + websocket-client`.

```bat
python -m badminton_analysis.live --source browser_headless --url "https://example.com/live" --fps 30 --display true
```

Prerequisite: Google Chrome installed (default `C:\Program Files\Google\Chrome\Application\chrome.exe`; override with `--chrome-path`).

### Stop and output

`Ctrl+C` to stop gracefully. Output goes to `outputs/live_<source>_<timestamp>/detect_*.mp4` (no audio, same path as `keep_audio=false`).

### Known limits

- Output mp4 fps is fixed to `--fps` (default 30); mismatched source rate changes playback speed
- The first frame is the court template source and appears at the head of the output video
- Override `--chrome-path` when Chrome is not at the default location
- macOS / Linux screen capture is untested (mss supports them but not verified here)
```

- [ ] **Step 5: Commit**

```powershell
git add README.md README_EN.md
git commit -m "docs: document real-time sources (mss + headless browser)"
```

---
