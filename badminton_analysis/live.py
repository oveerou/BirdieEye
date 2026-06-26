"""Real-time source entry point for Good-Badminton.

Usage:
    # Screen capture with a preset region
    python -m badminton_analysis.live --source screen_capture --preset fullscreen
    python -m badminton_analysis.live --source screen_capture --preset center-720p
    python -m badminton_analysis.live --source screen_capture --region 100,100,1280,720

    # Preview the captured region (one frame, no analysis, exits)
    python -m badminton_analysis.live --source screen_capture --region 100,100,1280,720 --preview

    # Skip court annotation (use whole frame; works for any content)
    python -m badminton_analysis.live --source screen_capture --region 100,100,1280,720 --no-court

    # Headless browser
    python -m badminton_analysis.live --source browser_headless --url "https://example.com/live"
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import cv2
import mss

from .sources import HeadlessBrowserSource, ScreenCaptureSource, StreamAdapter


REGION_PRESETS = {
    "fullscreen": None,        # filled at runtime from mss
    "left-half": None,
    "right-half": None,
    "top-half": None,
    "bottom-half": None,
    "center-720p": None,
    "custom": None,            # user provides --region
}


def _detect_screen_size() -> tuple[int, int]:
    """Return (width, height) of the primary monitor via mss."""
    with mss.MSS() as sct:
        mon = sct.monitors[1]  # primary
        return int(mon["width"]), int(mon["height"])


def _resolve_preset(name: str, custom: list[int] | None) -> list[int]:
    if name == "custom":
        if custom is None:
            raise SystemExit("--preset=custom requires --region L T W H")
        return list(custom)
    sw, sh = _detect_screen_size()
    if name == "fullscreen":
        return [0, 0, sw, sh]
    if name == "left-half":
        return [0, 0, sw // 2, sh]
    if name == "right-half":
        return [sw // 2, 0, sw - sw // 2, sh]
    if name == "top-half":
        return [0, 0, sw, sh // 2]
    if name == "bottom-half":
        return [0, sh // 2, sw, sh - sh // 2]
    if name == "center-720p":
        w, h = 1280, 720
        left = max(0, (sw - w) // 2)
        top = max(0, (sh - h) // 2)
        return [left, top, w, h]
    raise SystemExit(f"unknown preset: {name}")


class _RegionAction(argparse.Action):
    """Parse --region as either 'L,T,W,H' (one arg) or 'L T W H' (four args)."""

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, str):
            values = [values]
        if len(values) == 1 and "," in values[0]:
            parts = [p.strip() for p in values[0].split(",")]
        else:
            parts = list(values)
        if len(parts) != 4:
            parser.error(
                f"argument {option_string}: expected 4 ints "
                f"(L T W H or L,T,W,H), got {len(parts)}: {values!r}"
            )
        try:
            ints = [int(p) for p in parts]
        except ValueError as e:
            parser.error(f"argument {option_string}: non-integer value: {e}")
        setattr(namespace, self.dest, ints)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    p.add_argument("--region", nargs="*", action=_RegionAction, default=None,
                   metavar="L T W H",
                   help="(screen_capture) custom region; required when --preset=custom")
    p.add_argument("--preset", default="fullscreen",
                   choices=["fullscreen", "left-half", "right-half", "top-half",
                            "bottom-half", "center-720p", "custom"],
                   help="(screen_capture) region preset; default: fullscreen")
    p.add_argument("--url", default="", help="(browser_headless) URL to open")
    p.add_argument("--chrome-path", default=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                   help="(browser_headless) chrome.exe path")
    p.add_argument("--browser-w", type=int, default=1280)
    p.add_argument("--browser-h", type=int, default=720)
    p.add_argument("--wait-sec", type=float, default=10.0)
    p.add_argument("--interactive", action="store_true",
                   help="Show OpenCV court annotation window (default: auto-accept non-interactively)")
    p.add_argument("--no-court", action="store_true",
                   help="Skip court annotation entirely; analyze the whole frame as-is")
    p.add_argument("--preview", action="store_true",
                   help="Grab one frame and save it, then exit (no analysis)")
    p.add_argument("--preview-out", default=None,
                   help="Path to save preview frame (default: outputs/preview_<timestamp>.png)")
    return p.parse_args(argv)


def build_source(args: argparse.Namespace):
    if args.source == "screen_capture":
        region = _resolve_preset(args.preset, args.region)
        left, top, width, height = region
        return ScreenCaptureSource(left=left, top=top, width=width, height=height), region
    if args.source == "browser_headless":
        src = HeadlessBrowserSource(
            url=args.url,
            chrome_path=args.chrome_path,
            width=args.browser_w,
            height=args.browser_h,
            wait_sec=args.wait_sec,
        )
        return src, None
    raise ValueError(f"Unknown source: {args.source}")


def do_preview(args: argparse.Namespace) -> int:
    """Grab a single frame and save to a PNG. No analysis, no models loaded."""
    source, region = build_source(args)
    if not source.open():
        print(f"[preview] failed to open {args.source}", file=sys.stderr)
        return 2
    res = source.next_frame()
    source.close()
    if not res.ok or res.frame is None:
        print(f"[preview] failed to grab frame: {res.error}", file=sys.stderr)
        return 2
    if args.preview_out:
        out_path = args.preview_out
    else:
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", f"preview_{int(time.time())}.png")
    cv2.imwrite(out_path, res.frame)
    h, w = res.frame.shape[:2]
    print(f"[preview] saved {out_path} ({w}x{h})")
    if region is not None:
        print(f"[preview] region: left={region[0]} top={region[1]} {region[2]}x{region[3]}")
    return 0


def main() -> None:
    args = parse_args()

    if args.preview:
        raise SystemExit(do_preview(args))

    from .system import BadmintonAnalysisSystem, load_runtime_dependencies
    load_runtime_dependencies()

    run_id = f"live_{args.source}_{time.strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join("outputs", run_id)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[live] run_id={run_id}")
    print(f"[live] opening source: {args.source}")
    source, region = build_source(args)
    if region is not None:
        print(f"[live] region: left={region[0]} top={region[1]} {region[2]}x{region[3]}")
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
        non_interactive_annotation=not args.interactive,
        skip_court_annotation=args.no_court,
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
