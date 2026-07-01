import ast
import os
import queue
import tempfile
from tkinter import filedialog
import tkinter as tk
import time
import argparse


def load_runtime_dependencies():
    """Load heavy runtime dependencies after argparse has handled --help."""
    global cv2, np, YOLO, CourtMapper, annotate_court, compute_expanded_roi, PlayerTracker
    global CourtTrajectoryVisualizer, ShuttlecockTracker
    global PlayerPoseVisualizer, StatsVisualizer, RTMPoseProcessor, YOLOPoseProcessor, vap
    global JsonlDetectionWriter, write_json, SCHEMA_VERSION

    yolo_config_dir = os.path.join(tempfile.gettempdir(), "BirdieEye-ultralytics")
    os.makedirs(yolo_config_dir, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", yolo_config_dir)

    # GPU performance hints
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("medium")  # use TF32 on Ampere+
    except Exception:
        pass

    try:
        import cv2 as _cv2
        import numpy as _np
        from ultralytics import YOLO as _YOLO
        from .court.mapper import CourtMapper as _CourtMapper, annotate_court as _annotate_court
        from .court.mapper import compute_expanded_roi as _compute_expanded_roi
        from .tracking.player import PlayerTracker as _PlayerTracker
        from .visualization.court_trajectory import CourtTrajectoryVisualizer as _CourtTrajectoryVisualizer
        from .detection.shuttlecock import ShuttlecockTracker as _ShuttlecockTracker
        from .visualization.player_pose import PlayerPoseVisualizer as _PlayerPoseVisualizer
        from .visualization.stats import StatsVisualizer as _StatsVisualizer
        from .detection.rtmpose import RTMPoseProcessor as _RTMPoseProcessor
        from .detection.yolo_pose import YOLOPoseProcessor as _YOLOPoseProcessor
        from .media import video_audio as _vap
        from .data.writer import JsonlDetectionWriter as _JsonlDetectionWriter
        from .data.writer import write_json as _write_json
        from .data.writer import SCHEMA_VERSION as _SCHEMA_VERSION
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing Python dependency: {exc.name}. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    cv2 = _cv2
    np = _np
    YOLO = _YOLO
    CourtMapper = _CourtMapper
    annotate_court = _annotate_court
    compute_expanded_roi = _compute_expanded_roi
    PlayerTracker = _PlayerTracker
    CourtTrajectoryVisualizer = _CourtTrajectoryVisualizer
    ShuttlecockTracker = _ShuttlecockTracker
    PlayerPoseVisualizer = _PlayerPoseVisualizer
    StatsVisualizer = _StatsVisualizer
    RTMPoseProcessor = _RTMPoseProcessor
    YOLOPoseProcessor = _YOLOPoseProcessor
    vap = _vap
    JsonlDetectionWriter = _JsonlDetectionWriter
    write_json = _write_json
    SCHEMA_VERSION = _SCHEMA_VERSION

class BadmintonAnalysisSystem:
    def __init__(self, video_path, show_display=True,
                 show_skeletons=True, show_player_trajectories=True,
                 show_court_trajectory=True, show_shuttlecock_trajectory=True,
                 show_player_stats=True, show_performance_stats=False,
                 save_images=False, language='zh', output_dir=None,
                 ball_model_path='weights/yolo11s-ball.pt', template_path=None,
                 pose_mode='balanced', pose_family='rtmpose',
                 yolo_pose_model='yolo11n-pose.pt', show_pose_roi=False,
                 frame_source=None, non_interactive_annotation=False,
                 skip_court_annotation=False, device=None,
                 court_update_interval=8.0, court_update_min_quality=0.5,
                 enable_court_updater=False, enable_drift_corrector=False,
                 save_video=True):
        self.video_path = video_path
        self.show_display = show_display
        self.language = language
        self.template_path = template_path
        self.ball_model_path = ball_model_path
        self.pose_mode = pose_mode
        self.pose_family = pose_family
        self.yolo_pose_model = yolo_pose_model
        self.show_pose_roi = show_pose_roi
        self.frame_source = frame_source
        self.non_interactive_annotation = non_interactive_annotation
        self.skip_court_annotation = skip_court_annotation
        self.device = device
        self.court_update_interval = float(court_update_interval)
        self.court_update_min_quality = float(court_update_min_quality)
        self.enable_court_updater = bool(enable_court_updater)
        self.enable_drift_corrector = bool(enable_drift_corrector)
        self.save_video = bool(save_video)
        self.display_queue: "queue.Queue | None" = None
        # Court state, populated after _setup_court_annotation
        self.court_corners: list | None = None
        self.court_roi_corners: list | None = None
        self.mid_height: int | None = None
        self.court_mapper = None
        self.drift_corrector = None


        self.show_skeletons = show_skeletons
        self.show_player_trajectories = show_player_trajectories
        self.show_court_trajectory = show_court_trajectory
        self.show_shuttlecock_trajectory = show_shuttlecock_trajectory
        self.show_player_stats = show_player_stats
        self.show_performance_stats = show_performance_stats
        self.save_images = save_images  

        if self.frame_source is None and not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"Input video not found: {self.video_path}\n"
                "Pass a valid video file with --video-path."
            )
        if not os.path.exists(self.ball_model_path):
            raise FileNotFoundError(
                f"Ball detection model not found: {self.ball_model_path}\n"
                "Download or train a YOLO shuttlecock model and place it at "
                "weights/yolo11s-ball.pt, or pass its path with --ball-model."
            )
        
        if self.pose_family == 'yolo-pose':
            self.rtmpose_processor = YOLOPoseProcessor(
                model_path=self.yolo_pose_model, device=self.device,
            )
        else:
            self.rtmpose_processor = RTMPoseProcessor(mode=self.pose_mode, pose_family=self.pose_family)
        self.yolo_ball_model = YOLO(self.ball_model_path)

        # In-play court model updater (initialized in
        # process_video after _setup_court_annotation has set court_corners).
        self.court_updater = None

        self.last_stats_update_frame = 0

        self.video_name = os.path.basename(self.video_path)[:-4]
        self.save_dir = output_dir or os.path.join('outputs', self.video_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.images_save_dir = os.path.join(self.save_dir, 'detect_images')
        os.makedirs(self.images_save_dir, exist_ok=True)
        

        self.metadata_path = os.path.join(self.save_dir, "metadata.json")
        self.detections_path = os.path.join(self.save_dir, "detections.jsonl")
        self.output_video_path = os.path.join(self.save_dir, f"detect_{self.video_name}.mp4")
        self.detection_writer = None
        

        self.player_1_hand = "right"  
        self.player_2_hand = "right"  
        self.start_time = None
        self.end_time = None
        

        self.shuttlecock_tracker = ShuttlecockTracker(
            yolo_ball_model=self.yolo_ball_model,
            trajectory_length=15,
            show_trajectory=self.show_shuttlecock_trajectory,
            show_performance_stats=False
        )
        
        self.player_pose_visualizer = PlayerPoseVisualizer(
            rtmpose_processor=self.rtmpose_processor,
            show_skeletons=self.show_skeletons,
            show_player_trajectories=self.show_player_trajectories,
            show_performance_stats=False
        )
        

        self.court_trajectory_visualizer = CourtTrajectoryVisualizer()
        

        self.stats_update_interval_frames = 0
        self.cached_movement_stats = {}

        self.is_court_view_count = 0
        self.consecutive_non_court_frames = 0
        self.rally_active = False
        self.rally_count = 0  
        self.fps = 30  
        self.court_view_frames_threshold = 5
        self.non_court_frames_threshold = 5

        self.frame_width = 0
        self.frame_height = 0
        self.performance_log_interval_frames = 150
        self.total_frames = 0
        self.debug = False
    def process_video(self, stop_event=None):
        """Process the input video."""
        if self.debug:
            import datetime
            import traceback as _tb
            _log_path = os.path.join(self.save_dir, "debug_process_video.log")
            log_file = open(_log_path, "a", encoding="utf-8")
            log_file.write(f"\n[{datetime.datetime.now()}] ===== process_video CALLED =====\n")
            log_file.flush()
        else:
            log_file = open(os.devnull, "w")
            import traceback as _tb

        self.start_time = time.time()

        cap = self.frame_source if self.frame_source is not None else cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            log_file.write(f"[{datetime.datetime.now()}] EXIT: cannot open video source\n")
            log_file.flush()
            log_file.close()
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            log_file.write(f"[{datetime.datetime.now()}] EXIT: fps<=0 ({fps})\n")
            log_file.flush()
            log_file.close()
            raise RuntimeError(f"Unable to read FPS from video: {self.video_path}")
        if total_frames > 0:
            video_duration = total_frames / fps
        else:
            video_duration = 0
        log_file.write(f"[{datetime.datetime.now()}] source opened: fps={fps}, total_frames={total_frames}, duration={video_duration:.2f}s\n")
        log_file.flush()

        self.fps = fps
        self.performance_log_interval_frames = max(1, int(fps * 5))
        

        frame_count = 0
        detect_frame_count = 0
        out = None
        corners = None

        try:
            template_path = self._get_template_path()
            log_file.write(f"[{datetime.datetime.now()}] init: loading template {template_path}\n"); log_file.flush()
            template_gray, template_color = self._load_template(template_path, cap)
            log_file.write(f"[{datetime.datetime.now()}] init: template loaded OK\n"); log_file.flush()

            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.save_video:
                out = self._setup_video_writer(self.frame_width, self.frame_height, fps)
            else:
                out = None
                self.video_writer = None
                try:
                    print(f"[video] save_video=False, skipping MP4 output")
                except OSError:
                    pass

            log_file.write(f"[{datetime.datetime.now()}] init: setting up court annotation\n"); log_file.flush()
            corners, roi_corners, mid_height = self._setup_court_annotation(template_color)
            self.court_corners = corners
            self.court_roi_corners = roi_corners
            log_file.write(f"[{datetime.datetime.now()}] init: court annotation OK, corners={corners}\n"); log_file.flush()

            self._write_metadata(fps, total_frames, video_duration, template_path, corners, roi_corners, mid_height)
            self.detection_writer = JsonlDetectionWriter(self.detections_path)

            self.court_mapper = CourtMapper(corners)
            self.player_pose_visualizer.court_mapper = self.court_mapper
            log_file.write(f"[{datetime.datetime.now()}] init: creating PlayerTracker\n"); log_file.flush()
            self.player_tracker = PlayerTracker(corners=corners, threshold=mid_height, history_size=20,
                                              detection_writer=self.detection_writer, fps=fps)

            self.stats_visualizer = StatsVisualizer(
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                language=self.language
            )
            log_file.write(f"[{datetime.datetime.now()}] init: ALL INIT DONE, entering main loop\n"); log_file.flush()
        except Exception as e:
            log_file.write(f"[{datetime.datetime.now()}] EXIT: init Exception: {e}\n")
            log_file.write(_tb.format_exc())
            log_file.flush()
            log_file.close()
            raise

        # For live sources, never exit due to read failures - only manual stop
        # For video files, exit immediately on read failure (end of file)
        is_live = self.frame_source is not None
        max_consecutive_failures = 999999 if is_live else 1
        consecutive_failures = 0

        try:
            while cap.isOpened():
                # Check stop signal from UI
                if stop_event is not None and stop_event.is_set():
                    log_file.write(f"[{datetime.datetime.now()}] EXIT: stop_event is set at frame {frame_count}\n")
                    log_file.flush()
                    break
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        log_file.write(f"[{datetime.datetime.now()}] EXIT: max_consecutive_failures reached ({max_consecutive_failures}) at frame {frame_count}\n")
                        log_file.flush()
                        break
                    time.sleep(0.05)
                    continue
                consecutive_failures = 0
                frame_count += 1
                frame, detect_frame_count = self._process_frame(frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count)
            else:
                log_file.write(f"[{datetime.datetime.now()}] EXIT: cap.isOpened() returned False at frame {frame_count}\n")
                log_file.flush()
        except KeyboardInterrupt:
            log_file.write(f"[{datetime.datetime.now()}] EXIT: KeyboardInterrupt at frame {frame_count}\n")
            log_file.flush()
        except Exception as e:
            log_file.write(f"[{datetime.datetime.now()}] EXIT: Exception at frame {frame_count}: {e}\n")
            log_file.write(_tb.format_exc())
            log_file.flush()
        finally:
            self.total_frames = frame_count
            self.end_time = time.time()
            processing_time = self.end_time - self.start_time

            try:
                print(f"\n处理完成:")
                print(f"原始视频时长: {video_duration:.2f} 秒")
                print(f"处理耗时: {processing_time:.2f} 秒")
                if video_duration > 0:
                    print(f"处理速度比: {processing_time/video_duration:.2f}x")
                else:
                    print(f"处理速度比: N/A (live source, duration unknown)")
            except OSError:
                pass

            log_file.write(f"[{datetime.datetime.now()}] FINALLY: total_frames={frame_count}, closing\n")
            log_file.flush()
            try:
                log_file.close()
            except Exception:
                pass
            self._cleanup(cap)

    def get_summary(self) -> dict:
        """Collect actual metrics after process_video() completes.

        Returns a dict with the same keys that RunMetrics expects
        (total_frames, avg_fps, avg_player_count, ball_visible_ratio,
        upper_avg_speed, lower_avg_speed, total_rallies).
        """
        processing_time = 0.0
        if self.start_time and self.end_time:
            processing_time = self.end_time - self.start_time
        avg_fps = round(self.total_frames / processing_time, 2) if processing_time > 0 else 0.0

        upper_avg_speed = 0.0
        lower_avg_speed = 0.0
        stats = getattr(self, "cached_movement_stats", {}) or {}
        if "upper" in stats and isinstance(stats["upper"], dict):
            upper_avg_speed = float(stats["upper"].get("match_avg_speed", 0.0))
        if "lower" in stats and isinstance(stats["lower"], dict):
            lower_avg_speed = float(stats["lower"].get("match_avg_speed", 0.0))

        return {
            "total_frames": int(self.total_frames),
            "avg_fps": float(avg_fps),
            "avg_player_count": 0.0,
            "ball_visible_ratio": 0.0,
            "upper_avg_speed": float(upper_avg_speed),
            "lower_avg_speed": float(lower_avg_speed),
            "total_rallies": int(getattr(self, "rally_count", 0)),
        }

    def _write_metadata(self, fps, total_frames, video_duration, template_path, corners, roi_corners, mid_height):
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "video": {
                "path": self.video_path,
                "name": self.video_name,
                "fps": float(fps),
                "total_frames": int(total_frames),
                "duration_sec": float(video_duration),
                "width": int(self.frame_width),
                "height": int(self.frame_height),
            },
            "models": {
                "shuttlecock": self.ball_model_path,
            },
            "court": {
                "template_path": template_path,
                "corners": corners,
                "roi_corners": roi_corners,
                "mid_height": mid_height,
                "coordinate_system": {
                    "unit": "meter",
                    "width": 6.1,
                    "length": 13.4,
                },
            },
            "outputs": {
                "video": self.output_video_path,
                "detections": self.detections_path,
            },
        }
        write_json(self.metadata_path, metadata)

    def _push_to_display_queue(self, frame, detect_frame_count):
        if self.display_queue is None or frame is None:
            return
        try:
            self.display_queue.put_nowait((frame, detect_frame_count))
        except queue.Full:
            try:
                self.display_queue.get_nowait()
                self.display_queue.put_nowait((frame, detect_frame_count))
            except Exception:
                pass

    def _process_frame(self, frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count):

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # frame = self.draw_court_roi(frame, corners, roi_corners)

        is_court = self.is_court_view(gray_frame, template_gray)
        
        # ── Rally state machine (runs on every frame) ──
        if is_court:
            self.is_court_view_count += 1
            self.consecutive_non_court_frames = 0
        else:
            self.consecutive_non_court_frames += 1
            self.is_court_view_count = 0

        if self.is_court_view_count >= self.court_view_frames_threshold and not self.rally_active:
            self.rally_active = True
            self.rally_count += 1
            self.player_tracker.start_new_rally()

        if self.consecutive_non_court_frames >= self.non_court_frames_threshold and self.rally_active:
            self.rally_active = False
            self.shuttlecock_tracker.clear_trajectory()

        # ── Early return for non-court frames: skip all detection/tracking ──
        if not is_court:
            fh, fw = frame.shape[:2]
            if self.show_player_stats and self.cached_movement_stats:
                cv2.putText(frame, "STATS PAUSED", (20, fh - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
            self._push_to_display_queue(frame, detect_frame_count)
            return frame, detect_frame_count

        detect_frame_count += 1

        # Use the latest roi_corners (may have been updated by CourtModelUpdater)
        current_roi = self.court_roi_corners if self.court_roi_corners else roi_corners
        x1, y1 = current_roi[0]
        x2, y2 = current_roi[1]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            # ROI is invalid (e.g., court updater produced bad corners), fall back to full frame
            roi = frame
            x1, y1 = 0, 0

        if self.show_pose_roi:
            cv2.rectangle(frame, current_roi[0], current_roi[1], (255, 0, 0), 2)
            cv2.putText(frame, "Pose ROI", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # ── Detection (only runs on court frames — non-court frames returned early above) ──
        pose_t0 = time.time()
        centroids, point_left_hands, point_right_hands = self.player_pose_visualizer.detect_players(roi, x1, y1)
        pose_elapsed = time.time() - pose_t0

        # ── Drift correction: periodic check + apply ──
        if hasattr(self, 'drift_corrector') and self.drift_corrector is not None:
            self.drift_corrector.maybe_check(frame)
            if centroids and self.drift_corrector.is_active:
                centroids = self.drift_corrector.correct(centroids)

        ball_t0 = time.time()
        detected_ball_position = self.shuttlecock_tracker.detect_ball(frame, roi_corners=current_roi)
        ball_elapsed = time.time() - ball_t0
        ball_position = self.shuttlecock_tracker.update_trajectory(detected_ball_position, current_roi)

        shuttle_draw_t0 = time.time()
        self.shuttlecock_tracker.handle_visualization(frame)
        shuttle_draw_elapsed = time.time() - shuttle_draw_t0

        # ── Tracking ──
        players = self.player_tracker.update(frame_count, centroids, ball_position,
                                             point_left_hands, point_right_hands, detect_frame_count)

        # ── CourtModelUpdater ──
        if self.court_updater is not None:
            try:
                self.court_updater.maybe_update(frame, len(centroids))
            except Exception:
                pass

        # ── Stats ──
        if frame_count == 1 or not self.cached_movement_stats:
            self.cached_movement_stats = self.player_tracker.get_player_movement_stats()
            self.stats_update_interval_frames = int(self.player_tracker.fps * 0.5)

        if frame_count - self.last_stats_update_frame >= self.stats_update_interval_frames:
            self.cached_movement_stats = self.player_tracker.get_player_movement_stats()
            self.last_stats_update_frame = frame_count


        should_log_performance = (
            self.show_performance_stats
            and self.performance_log_interval_frames > 0
            and frame_count % self.performance_log_interval_frames == 0
        )

        t0 = time.time()

        active_stats_viz = self.stats_visualizer if self.show_player_stats else None
        self.player_pose_visualizer.draw_players(
            frame=frame,
            player_tracker=self.player_tracker,
            cached_movement_stats=self.cached_movement_stats,
            stats_visualizer=active_stats_viz,
            rally_count=self.rally_count
        )

        t1 = time.time()
        players_draw_elapsed = t1 - t0
        

        court_draw_elapsed = 0.0
        if self.show_court_trajectory:
            t0 = time.time()
            frame = self.court_trajectory_visualizer.draw_overlay(frame, self.player_tracker.court_history)
            t1 = time.time()
            court_draw_elapsed = t1 - t0

        if should_log_performance:
            print(
                f"Frame {frame_count}: pose {pose_elapsed:.2f}s, "
                f"shuttlecock {ball_elapsed:.2f}s, "
                f"shuttle draw {shuttle_draw_elapsed:.2f}s, "
                f"players draw {players_draw_elapsed:.2f}s, "
                f"court draw {court_draw_elapsed:.2f}s"
            )
        

        if frame is not None:
            if self.show_display:
                cv2.imshow('frame', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
                    raise KeyboardInterrupt
            if out is not None:
                out.write(frame)

            if self.save_images:
                cv2.imwrite(os.path.join(self.images_save_dir, f"{frame_count}.png"), frame)
        self._push_to_display_queue(frame, detect_frame_count)
        return frame, detect_frame_count

    def process_frame(self, frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count):
        """Public alias for _process_frame, used by the Streamlit Web UI."""
        return self._process_frame(frame, template_gray, corners, roi_corners, frame_count, out, detect_frame_count)

    def _get_template_path(self):
        """Get the court template image path."""
        if self.template_path:
            if not os.path.exists(self.template_path):
                raise FileNotFoundError(
                    f"Court template image not found: {self.template_path}"
                )
            return self.template_path

        try:
            root = tk.Tk()
            root.withdraw()
            template_path = filedialog.askopenfilename(
                title="Select court template image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
            )
            root.destroy()
        except Exception as exc:
            raise RuntimeError(
                "Unable to open the template picker. In headless environments, "
                "pass a court template image path with --template-path."
            ) from exc

        if not template_path:
            raise RuntimeError(
                "No court template image selected. Pass --template-path to run "
                "without the file picker."
            )
        return template_path

    def _load_template(self, template_path, cap):
        """Load and resize the court template image."""
        template_gray = cv2.imread(template_path, 0)
        template_color = cv2.imread(template_path)
        if template_gray is None or template_color is None:
            raise RuntimeError(f"Unable to read court template image: {template_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        template_gray = cv2.resize(template_gray, (frame_width, frame_height))
        template_color = cv2.resize(template_color, (frame_width, frame_height))
        
        return template_gray, template_color

    def _setup_video_writer(self, frame_width, frame_height, fps):

        self.temp_output_video_path = os.path.join(self.save_dir, f"temp_detect_{self.video_name}.mp4")
        

        self.video_writer = vap.setup_video_writer(
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            temp_output_path=self.temp_output_video_path
        )
        
        return self.video_writer

    def _setup_court_annotation(self, template_color):
        """Set up court annotation."""

        if getattr(self, "skip_court_annotation", False):
            h, w = template_color.shape[:2]
            corners = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
            roi_corners = [(0, 0), (w - 1, h - 1)]
            mid_height = h // 2
            print(f"[court] --no-court: skipping annotation, using full frame "
                  f"({w}x{h}) with horizontal mid_height={mid_height}")
            return corners, roi_corners, mid_height

        if os.path.exists(os.path.join(self.save_dir, 'court_annotations.txt')):
            with open(os.path.join(self.save_dir, 'court_annotations.txt'), 'r') as f:
                corners = ast.literal_eval(f.readline().split('=', 1)[1])
                f.readline()
                mid_height = ast.literal_eval(f.readline().split('=', 1)[1])
                roi_corners = compute_expanded_roi(corners, template_color.shape)
        else:
            auto_preview_path = os.path.join(self.save_dir, 'auto_court_preview.png')
            corners, roi_corners, mid_height = annotate_court(
                template_color,
                auto_preview_path=auto_preview_path,
                non_interactive=self.non_interactive_annotation,
            )
       
        if not corners or not roi_corners or len(corners) != 4 or len(roi_corners) != 2:
            raise RuntimeError("Court annotation is incomplete: click 4 court corners in order. ROI is generated automatically.")

        with open(os.path.join(self.save_dir, 'court_annotations.txt'), 'w') as f:
            f.write(f"corners={corners}\n")
            f.write(f"roi_corners={roi_corners}\n")
            f.write(f"mid_height={mid_height}\n")
        # Cache on the system so the updater / display pipeline can read them.
        self.court_corners = list(corners)
        self.court_roi_corners = list(roi_corners)
        self.mid_height = int(mid_height)
        # Initialize the in-play updater now that we have a model.
        from .court.updater import CourtModelUpdater
        from .court.drift_corrector import CourtDriftCorrector
        if self.enable_court_updater:
            self.court_updater = CourtModelUpdater(
                self,
                check_interval_sec=self.court_update_interval,
                min_quality=self.court_update_min_quality,
            )
        else:
            self.court_updater = None

        # Drift corrector: periodically re-detect court corners and compensate
        # for camera movement.  Disabled by default — auto_detect_court_corners
        # is unreliable on broadcast footage (HSV finds green patches, not court
        # lines), which produces wrong homographies that shift centroids off-target.
        if self.enable_drift_corrector:
            self.drift_corrector = CourtDriftCorrector(corners, check_interval=90)
        else:
            self.drift_corrector = None
        return corners, roi_corners, mid_height

    def _cleanup(self, cap):
        """Clean up resources and merge audio when needed."""
        if self.detection_writer is not None:
            self.detection_writer.close()
            self.detection_writer = None

        if self.save_video:
            if hasattr(self, 'video_writer') and self.video_writer is not None:
                self.video_writer.release()
                time.sleep(1)

        cap.release()

        if self.show_display:
            cv2.destroyAllWindows()

        if not self.save_video:
            return

        if hasattr(self, 'keep_audio') and self.keep_audio:
            vap.process_video_with_audio(
                video_path=self.video_path,
                temp_video_path=self.temp_output_video_path,
                output_path=self.output_video_path,
                save_dir=self.save_dir
            )
        else:
            vap.process_video_without_audio(
                temp_video_path=self.temp_output_video_path,
                output_path=self.output_video_path
            )

    def is_court_view(self, frame, template_gray, threshold=0.70):
        """Return whether the frame matches the court template.

        Threshold 0.55 balances precision and recall for real-time screen-capture
        sources. The original offline pipeline uses 0.75, but captured frames may
        have resolution/compression differences that lower the raw score slightly.
        Values below ~0.50 admit too many non-court frames (close-ups, replays)
        which feed garbage positions into the tracker and heatmap.
        """
        # Cache the result for a few frames to reduce CPU overhead.
        # Template matching is expensive and court view status changes slowly.
        cache_count = getattr(self, '_is_court_cache_count', 0)
        if cache_count > 0 and hasattr(self, '_is_court_cache_result'):
            self._is_court_cache_count = cache_count - 1
            return self._is_court_cache_result

        result = cv2.matchTemplate(frame, template_gray, cv2.TM_CCOEFF_NORMED)
        score = float(np.max(result))
        is_court = score >= threshold
        if not getattr(self, '_is_court_debug_printed', False):
            try:
                print(f"[court] is_court_view: first match score = {score:.4f} (threshold={threshold})")
            except OSError:
                pass
            self._is_court_debug_printed = True
        self._is_court_cache_result = is_court
        self._is_court_cache_count = 5  # reuse for next 5 frames
        return is_court

    def draw_court_roi(self, frame, corners, roi_corners):
        self.court_mapper = CourtMapper(corners)
        overlay, mid_height_int = self.court_mapper.draw_court_overlay(frame)
        cv2.rectangle(overlay, roi_corners[0], roi_corners[1], (255, 0, 0), 2)
        return overlay
