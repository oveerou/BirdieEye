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
    p.add_argument("--interactive", action="store_true",
                   help="Show OpenCV court annotation window (default: auto-accept non-interactively)")
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
        non_interactive_annotation=not args.interactive,
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
