# Add Real-Time Sources to Good-Badminton (mss + Headless Browser)

- **Date**: 2026-06-27
- **Status**: Draft
- **Scope**: Bring two video sources from `D:\A_shixi\football-realtime-analyzer` into `D:\A_shixi\Good-Badminton` without breaking the existing local-file workflow.

## 1. Background

`Good-Badminton` currently only accepts local video files via `--video-path` (validated by `os.path.exists` in `badminton_analysis/system.py:89` and opened with `cv2.VideoCapture(self.video_path)` in `badminton_analysis/system.py:166`). All downstream analysis (court annotation, pose detection, ball tracking, statistics, output video) assumes:

- A file on disk
- Known FPS, frame count, and width/height available via `cap.get(...)` *before* the main loop
- An audio track that can be merged back via `moviepy` / `ffmpeg` (`vap.process_video_with_audio` in `badminton_analysis/system.py:480-486`)

`football-realtime-analyzer` (`D:\A_shixi\football-realtime-analyzer`) demonstrates two non-file sources we want to add:

1. **`ScreenCaptureSource`** — captures a screen region with `mss` (no FPS, infinite stream, no audio, dimensions known only after first grab).
2. **`HeadlessBrowserSource`** — launches Chrome `--headless=new` with remote debugging and pulls `<video>` frames via Chrome DevTools Protocol (CDP) over WebSocket. Same characteristics as mss.

Both implement a common `FrameSource` abstraction in `football_analyzer/sources/` (`base.py` / `screen_capture.py` / `headless_browser.py`) with `open()` / `next_frame()` / `close()` and a `FrameResult` dataclass.

## 2. Goal

Allow a user to run Good-Badminton's full analysis pipeline on:

- A screen region (e.g., a video player window open in a browser)
- A webpage containing a `<video>` element (live stream, replay page)

…with no changes to the existing `python main.py --video-path xxx.mp4` flow and no rewrites of `system.py`'s analysis logic.

## 3. Non-Goals

- Refactoring `process_video()` into a streaming pipeline.
- Adding per-source CLI flags inside `main.py` (kept as a separate entry point).
- Replacing `cv2.VideoCapture` everywhere.
- Supporting macOS/Linux screen capture edge cases (mss already handles this; we only need to verify on Windows).
- Audio capture (neither source has audio).
- Real-time metrics streaming / WebSocket UI (the existing project writes results to disk; we keep that).
- Fine-tuning YOLO models or annotating new training data.

## 4. Design

### 4.1 Architecture

```
badminton_analysis/
├── system.py            ← MODIFIED: +1 optional __init__ param, +1 branch in process_video
├── live.py              ← NEW: entry point for real-time sources
├── sources/             ← NEW: copied from football + adapter
│   ├── __init__.py
│   ├── base.py
│   ├── screen_capture.py
│   ├── headless_browser.py
│   └── stream_adapter.py
└── ...

main.py                  ← UNCHANGED
requirements.txt         ← +mss, +websocket-client
README.md                ← +"Real-time sources" section
```

### 4.2 New component: `StreamAdapter`

Location: `badminton_analysis/sources/stream_adapter.py`

Wraps a `FrameSource` to expose the subset of `cv2.VideoCapture` interface that `BadmintonAnalysisSystem.process_video()` actually uses.

```python
class StreamAdapter:
    def __init__(
        self,
        source: "FrameSource",
        fps: float = 30.0,
        first_frame: np.ndarray | None = None,
    ):
        self._source = source
        self._fps = fps
        self._first_frame = first_frame   # pre-buffered frame to be served first
        self._opened = True

    def isOpened(self) -> bool: return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._first_frame is not None:
            frame = self._first_frame
            self._first_frame = None      # consume
            return True, frame
        res = self._source.next_frame()
        if not res.ok:
            return False, None
        return True, res.frame

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            frame = self._first_frame
            if frame is None:
                # Fall back to a fresh grab (best-effort; should not normally hit)
                res = self._source.next_frame()
                if res.ok and res.frame is not None:
                    self._first_frame = res.frame
                    frame = res.frame
            if frame is None:
                return 0
            return float(frame.shape[1] if prop == cv2.CAP_PROP_FRAME_WIDTH else frame.shape[0])
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return -1.0   # unknown for live streams
        return 0.0

    def release(self) -> None:
        if self._opened:
            self._source.close()
            self._opened = False
```

**Rationale**:

- Keeps `process_video()` ignorant of whether `cap` is a file or stream.
- Pre-buffering `first_frame` lets `get(CAP_PROP_FRAME_WIDTH/HEIGHT)` return real dimensions without a side effect that would consume the first analyzed frame.
- `CAP_PROP_FRAME_COUNT = -1` is `cv2`'s convention for "unknown"; `process_video()` divides by it on line 174 (`video_duration = total_frames / fps`); we will guard that division in `process_video()` (see §4.3).

### 4.3 Modified component: `BadmintonAnalysisSystem`

`badminton_analysis/system.py` — two minimal changes.

**Change 1**: `__init__` (around line 70) — add one optional parameter.

```python
def __init__(self, video_path, show_display=True,
             show_skeletons=True, show_player_trajectories=True,
             show_court_trajectory=True, show_shuttlecock_trajectory=True,
             show_player_stats=True, show_performance_stats=False,
             save_images=False, language='zh', output_dir=None,
             ball_model_path='weights/yolo11s-ball.pt', template_path=None,
             pose_mode='balanced', pose_family='rtmpose',
             yolo_pose_model='yolo11n-pose.pt', show_pose_roi=True,
             frame_source=None):                                  # ← NEW
    ...
    self.frame_source = frame_source                            # ← NEW
```

**Change 2**: `process_video()` (around lines 166-174) — two small changes (one line replacement, one guard).

```python
def process_video(self):
    self.start_time = time.time()
    cap = self.frame_source if self.frame_source is not None else cv2.VideoCapture(self.video_path)   # ← CHANGED (line 166)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {self.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise RuntimeError(f"Unable to read FPS from video: {self.video_path}")
    if total_frames > 0:                                        # ← CHANGED (guard for live source around line 174)
        video_duration = total_frames / fps
    else:
        video_duration = 0                                      # live source: unknown duration
    ...
```

**Rationale**:

- `frame_source=None` (default) preserves existing behavior bit-for-bit; `main.py` does not need to change.
- The `isOpened()` call works for both `cv2.VideoCapture` and `StreamAdapter` (both return bool), so the existing check line is unchanged.
- The `total_frames > 0` guard prevents `ZeroDivisionError` when `StreamAdapter.get(CAP_PROP_FRAME_COUNT)` returns -1 (live stream) or when a corrupted file returns 0.
- Everything else in `process_video()` (`_get_template_path`, `_load_template`, `_setup_court_annotation`, main loop, `_cleanup`) works unchanged for both paths.

**Note on audio**: The existing `_cleanup` (line 465-489) already has an `if self.keep_audio` branch. `live.py` sets `keep_audio=False` on the system (no audio for live sources), so the existing `process_video_without_audio` (just `shutil.copy2`) runs unchanged.

### 4.4 New component: `live.py`

Location: `badminton_analysis/live.py`

```python
"""
Real-time source entry point for Good-Badminton.

Usage:
    python -m badminton_analysis.live --source screen_capture \
           --region 100,100,1280,720 --fps 30 --display true

    python -m badminton_analysis.live --source browser_headless \
           --url https://example.com/live --fps 30 --display true
"""

import argparse
import os
import shutil
import time

import cv2

from .sources import (
    HeadlessBrowserSource,
    ScreenCaptureSource,
    StreamAdapter,
)


def build_source(args) -> "FrameSource":
    if args.source == "screen_capture":
        return ScreenCaptureSource(
            left=args.region[0],
            top=args.region[1],
            width=args.region[2],
            height=args.region[3],
        )
    if args.source == "browser_headless":
        return HeadlessBrowserSource(
            url=args.url,
            chrome_path=args.chrome_path,
            width=args.browser_w,
            height=args.browser_h,
            wait_sec=args.wait_sec,
        )
    raise ValueError(f"Unknown source: {args.source}")


def parse_args():
    p = argparse.ArgumentParser(description="Good-Badminton real-time source")
    p.add_argument("--source", required=True, choices=["screen_capture", "browser_headless"])
    p.add_argument("--fps", type=float, default=30.0, help="Output video fps")
    p.add_argument("--display", choices=["true", "false"], default="true")
    p.add_argument("--language", default="zh", choices=["zh", "en"])
    p.add_argument("--pose-family", default="yolo-pose", choices=["rtmpose", "rtmo", "yolo-pose"])
    p.add_argument("--pose-mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    p.add_argument("--yolo-pose-model", default="weights/yolo11n-pose.pt")
    p.add_argument("--ball-model", default="weights/yolo11s-ball.pt")
    p.add_argument("--display-overlay", choices=["true", "false"], default="true",
                   help="Whether to draw overlays (skeletons/trajectories)")
    # screen_capture specific
    p.add_argument("--region", type=int, nargs=4, default=[100, 100, 1280, 720],
                   metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    # browser_headless specific
    p.add_argument("--url", default="")
    p.add_argument("--chrome-path", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    p.add_argument("--browser-w", type=int, default=1280)
    p.add_argument("--browser-h", type=int, default=720)
    p.add_argument("--wait-sec", type=float, default=10.0)
    return p.parse_args()


def main():
    args = parse_args()
    # Lazy import to keep --help fast
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
    print(f"[live] source opened")

    print(f"[live] grabbing first frame (for court template)...")
    first = source.next_frame()
    if not first.ok or first.frame is None:
        source.close()
        raise SystemExit(f"[live] failed to grab first frame: {first.error}")

    first_frame_path = os.path.join(save_dir, "first_frame.png")
    cv2.imwrite(first_frame_path, first.frame)
    print(f"[live] first frame saved: {first_frame_path} ({first.frame.shape[1]}x{first.frame.shape[0]})")

    adapter = StreamAdapter(source=source, fps=args.fps, first_frame=first.frame)

    system = BadmintonAnalysisSystem(
        video_path=f"{run_id}.mp4",          # synthetic name; system will strip .mp4
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
    system.keep_audio = False               # live sources have no audio

    try:
        system.process_video()
        print(f"[live] finished: {system.output_video_path}")
    except KeyboardInterrupt:
        print(f"\n[live] interrupted by user (Ctrl+C)")
        # Best-effort finalize: release the in-progress writer and copy temp to output
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

### 4.5 New components: `sources/` package

| File | Action | Notes |
|---|---|---|
| `sources/base.py` | **Copy verbatim from** `D:\A_shixi\football-realtime-analyzer\football_analyzer\sources\base.py` | `FrameSource` ABC + `FrameResult` dataclass. No project-specific logic. |
| `sources/screen_capture.py` | **Copy verbatim from** `football_analyzer\sources\screen_capture.py` | `ScreenCaptureSource` using `mss`. |
| `sources/headless_browser.py` | **Copy verbatim from** `football_analyzer\sources\headless_browser.py` | `HeadlessBrowserSource` using subprocess + CDP over WebSocket. |
| `sources/stream_adapter.py` | New write, see §4.2 | Adapter exposing `cv2.VideoCapture`-shaped interface. |
| `sources/__init__.py` | New write | `from .base import FrameSource, FrameResult; from .screen_capture import ScreenCaptureSource; from .headless_browser import HeadlessBrowserSource; from .stream_adapter import StreamAdapter` |

**Dependency note**: `headless_browser.py` uses `subprocess` (stdlib), `base64` (stdlib), `json` (stdlib), `requests`, `websocket-client`, and `cv2` (already a project dep). The first three are stdlib; the last three need to be present in `requirements.txt`.

### 4.6 Requirements

Append to `requirements.txt`:

```
mss>=10.0.0
websocket-client
requests>=2.28
```

(`requests` is also pulled transitively by `ultralytics` / `moviepy`; pinning it explicitly keeps the headless browser source self-contained.)

### 4.7 README addition

Add a "Real-time sources" section to `README.md` after the existing "Usage" section, with:

- One-paragraph intro
- `mss` screen capture usage example
- `HeadlessBrowserSource` usage example with Chrome prerequisites
- Stop instructions: Ctrl+C
- Caveats: output fps is fixed to `--fps`; the first frame appears at the head of the output video (it is the same image used for court annotation, so the output starts with the annotated court view)

## 5. Data Flow

### 5.1 Existing flow (unchanged)

```
python main.py --video-path xxx.mp4
  → BadmintonAnalysisSystem(video_path=xxx.mp4, frame_source=None)
  → cap = cv2.VideoCapture(xxx.mp4)
  → ... unchanged ...
  → outputs/<video_name>/detect_*.mp4 + .jsonl + .json
```

### 5.2 New live flow

```
python -m badminton_analysis.live --source screen_capture --region 100,100,1280,720
  1. ScreenCaptureSource(...).open()                  ← mss init
  2. first = source.next_frame()                       ← grab first frame
  3. cv2.imwrite(outputs/live_*/first_frame.png, first.frame)   ← court template source
  4. adapter = StreamAdapter(source, fps=30, first_frame=first.frame)
  5. BadmintonAnalysisSystem(
       video_path="live_screen_capture_20260627_225800.mp4",
       template_path=outputs/live_*/first_frame.png,
       frame_source=adapter,
       keep_audio=False,
     )
  6. system.process_video()
       cap = adapter
       fps = cap.get(CAP_PROP_FPS) → 30
       total_frames = cap.get(CAP_PROP_FRAME_COUNT) → -1 → video_duration = 0
       template = first_frame (loaded via _load_template)
       corners, roi_corners, mid_height = annotate_court(template)   ← existing OpenCV window
       while cap.isOpened():
         ret, frame = cap.read()                    ← adapter serves buffered first_frame, then mss
         ... existing analysis ...
         out.write(frame)                           ← writes to temp mp4 at 30 fps
  7. Ctrl+C → KeyboardInterrupt
       adapter.release()
       shutil.copy2(temp_mp4, final_mp4)
  Final: outputs/live_*/detect_*.mp4 (overlaid, no audio)
```

### 5.3 Browser flow differences from §5.2

Steps 1-2 differ:

1. `HeadlessBrowserSource(url, chrome_path, ...).open()`:
   - `subprocess.Popen([chrome.exe, "--headless=new", ...])`
   - Poll `GET /json/list` until `webSocketDebuggerUrl` appears
   - `websocket.create_connection(ws_url)`
2. `first = source.next_frame()`:
   - `_grab_video_frame()` runs JS in Chrome to read `<video>` to canvas, base64 jpeg → BGR frame
   - Falls back to `_grab_screenshot()` (full page) if no `<video>` found

Steps 3-7 are identical to §5.2.

## 6. Error Handling

| Failure | Behavior |
|---|---|
| `mss` not installed | `ScreenCaptureSource.open()` returns False; `live.py` raises `SystemExit("failed to open screen_capture")`. Documented in README. |
| Screen region out of bounds | `mss` raises; `ScreenCaptureSource.next_frame()` returns `FrameResult.failure(...)`; main loop breaks naturally. |
| Chrome not found | `subprocess.Popen` raises `FileNotFoundError`; `HeadlessBrowserSource.open()` returns False; `live.py` exits. |
| Chrome CDP times out (`wait_sec`) | `HeadlessBrowserSource.open()` returns False after `wait_sec * 2` retries. `live.py` exits. |
| `<video>` element not on page | `_grab_video_frame()` returns None; `_grab_screenshot()` used as fallback. If both fail, `next_frame` returns failure, main loop ends. |
| User presses `q` in OpenCV display | `_process_frame` raises `KeyboardInterrupt`; handled in `live.py` like Ctrl+C. |
| User presses `X` on OpenCV window | Same as `q` (patch already in place from earlier work). |
| `pip install mss` missing on user's venv | `ModuleNotFoundError` at `import mss` inside `open()`; caught and returns False. |
| Output mp4 partial (interrupt) | `shutil.copy2` copies whatever was written; most players can play incomplete mp4 files. Documented caveat. |

## 7. Testing & Verification

Three smoke tests, in order of cost:

### 7.1 Static checks

```powershell
cd D:\A_shixi\Good-Badminton
.\.venv\Scripts\python.exe -c "from badminton_analysis.sources import StreamAdapter, ScreenCaptureSource, HeadlessBrowserSource; print('imports OK')"
.\.venv\Scripts\python.exe -c "from badminton_analysis import live; print('live module OK')"
```

### 7.2 Regression test: existing local file flow still works byte-for-byte

```powershell
# Clean previous outputs
Remove-Item outputs/demo1 -Recurse -Force
.\.venv\Scripts\python.exe main.py --video-path videos/demo1.mp4 --template-path templates/demo1.png --display false
```

Expected:
- `outputs/demo1/detect_demo1.mp4` is **22,112,015 bytes** (matches historical reference run).
- `detections.jsonl` is **467,583 bytes**.
- `metadata.json` is **944 bytes**.

If any size differs, the `process_video()` change broke something — fix before proceeding.

### 7.3 mss source smoke test

Pick a fixed screen region that contains a known visible window or pattern (e.g., a Notepad window with text). Run for ~10 seconds then Ctrl+C.

```powershell
# Bring Notepad or any visible window to the foreground first
.\.venv\Scripts\python.exe -m badminton_analysis.live --source screen_capture --region 100,100,1280,720 --fps 30 --display true
# ... wait 10s ...
# Press Ctrl+C in the terminal
```

Expected:
- OpenCV window opens showing live screen capture with pose/trajectory overlays.
- Court annotation prompt appears (auto-detect first; may require manual 4-click if auto fails).
- On Ctrl+C: `outputs/live_screen_capture_<timestamp>/detect_*.mp4` is created.
- File is non-zero size and playable.

### 7.4 Browser source smoke test

Find a public webpage with a `<video>` element. The football project's `app.py:80-94` test-connectivity example can be used. e.g., a Big Buck Bunny mirror, or a CCTV5 live page.

```powershell
.\.venv\Scripts\python.exe -m badminton_analysis.live --source browser_headless --url "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4" --fps 30 --display true
# ... wait 10s ...
# Ctrl+C
```

Expected:
- Chrome process spawns silently (no visible window).
- OpenCV window shows frames from the video.
- Output mp4 is generated on Ctrl+C.

## 8. Rollout

1. **Add files**: copy `base.py`, `screen_capture.py`, `headless_browser.py` from football; create `stream_adapter.py`, `sources/__init__.py`, `live.py`.
2. **Patch `system.py`**: add `frame_source=None` parameter; add `isOpened()` check; guard `total_frames > 0` for `video_duration`.
3. **Update `requirements.txt`**: add `mss`, `websocket-client`.
4. **Run §7.1 + §7.2** to verify no regression.
5. **Run §7.3** to verify mss source.
6. **Run §7.4** to verify browser source.
7. **Update `README.md`** with the new section.
8. **Commit** with message: `feat: add mss screen capture and headless browser real-time sources`.

## 9. Open Questions

None blocking. Implementation will discover the answer to:

- Does the test environment have Chrome installed at the default path? (verify in §7.4; fall back to `--chrome-path` if not)
- Will the user be able to use `HeadlessBrowserSource` with a CDN that serves CORS-restricted videos? (Not a blocker; if it doesn't work, the source falls back to `Page.captureScreenshot` of the whole page.)

## 10. References

- Source project: `D:\A_shixi\football-realtime-analyzer\football_analyzer\sources\`
- Target project: `D:\A_shixi\Good-Badminton\`
- OpenCV `VideoCapture` API: `cv2.CAP_PROP_FPS` / `CAP_PROP_FRAME_WIDTH` / `CAP_PROP_FRAME_HEIGHT` / `CAP_PROP_FRAME_COUNT`
- Chrome DevTools Protocol: `Page.captureScreenshot`, `Runtime.evaluate`
- mss: `mss.MSS().grab(region)` returns a `ScreenShot` with `.rgb` bytes
