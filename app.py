"""Streamlit Web UI for BirdieEye.

Mirrors the structure of football-realtime-analyzer/app.py
(sidebar + main + history) but uses badminton-native detection
(YOLO Pose + ball detection + court mapping + upper/lower player tracking).
"""
from __future__ import annotations

import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from badminton_analysis.config import (
    OUTPUTS_DIR,
    RUNS_DIR,
    UPLOADS_DIR,
    VIDEOS_DIR,
    load_config,
)
from badminton_analysis.sources import (
    HeadlessBrowserSource,
    ScreenCaptureSource,
    StreamAdapter,
    VideoFileSource,
)
from badminton_analysis.storage import (
    RunMetrics,
    init_db,
    insert_run,
    save_metrics_json,
)
from badminton_analysis.system import BadmintonAnalysisSystem, load_runtime_dependencies
from badminton_analysis.court.detector import auto_detect_court_corners
from badminton_analysis.court.mapper import compute_expanded_roi, CourtMapper
from badminton_analysis.analytics.post_analysis import run_post_analysis

# Eagerly load heavy runtime dependencies (YOLO model, etc.) once at app start,
# so that subsequent "开始识别" clicks don't take 5-10s to spin up the model.
with st.spinner("加载 YOLO 模型中（约 5-10 秒）..."):
    load_runtime_dependencies()

st.set_page_config(page_title="BirdieEye 实时识别系统", layout="wide")
st.title("基于 YOLO 的羽毛球比赛多源实时目标检测与分析系统")

with st.expander("使用前须知（点击展开）", expanded=False):
    st.markdown("""
    **首次使用请确认以下事项：**
    - **球场标定**：系统会自动检测球场边界，但一般不怎么准确，建议手动标注 4 个角点
    - **停止识别后**：全场数据统计和热力图需要一定时间生成，请耐心等待，完成后会自动显示在页面下方
    - **屏幕捕获**：请确保目标窗口未被遮挡，否则可能捕获失败
    - **GPU 加速**：系统默认使用 CUDA 加速，若无 GPU 可在侧边栏切换为 CPU 模式
    """)

# Fix: streamlit_image_coordinates needs the iframe to be large enough for
# click events to register reliably.  Without this, clicks may only work
# when the browser is in fullscreen (F11) mode.
st.markdown("""
<style>
  /* Ensure image-coordinate iframes fill their container and accept clicks */
  iframe[title="streamlit_image_coordinates.image_coordinates"] {
    width: 100% !important;
    min-height: 300px !important;
    pointer-events: auto !important;
  }
</style>
""", unsafe_allow_html=True)

cfg = load_config()

SOURCE_LABELS = {
    "video_file": "本地视频",
    "browser_headless": "网页直播（无头浏览器）",
    "screen_capture": "屏幕捕获",
}

PRESET_NAMES = ["全屏", "左半屏", "右半屏", "上半屏", "下半屏", "中央 1280x720", "自定义"]


def _detect_screen_size() -> tuple[int, int]:
    import mss
    with mss.MSS() as sct:
        mon = sct.monitors[1]
        return int(mon["width"]), int(mon["height"])


def _resolve_preset(name: str) -> list[int]:
    if name == "自定义":
        return [100, 100, 1280, 720]
    sw, sh = _detect_screen_size()
    if name == "全屏":
        return [0, 0, sw, sh]
    if name == "左半屏":
        return [0, 0, sw // 2, sh]
    if name == "右半屏":
        return [sw // 2, 0, sw - sw // 2, sh]
    if name == "上半屏":
        return [0, 0, sw, sh // 2]
    if name == "下半屏":
        return [0, sh // 2, sw, sh - sh // 2]
    if name == "中央 1280x720":
        w, h = 1280, 720
        return [max(0, (sw - w) // 2), max(0, (sh - h) // 2), w, h]
    return [100, 100, 1280, 720]


def _make_screen_source(region: list[int]) -> ScreenCaptureSource:
    return ScreenCaptureSource(
        left=region[0], top=region[1], width=region[2], height=region[3]
    )


with st.sidebar:
    st.header("参数设置")
    # GPU info (small, at the top so the user can see what's actually available)
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU only"
        st.caption(f"GPU: {gpu_name}" if cuda_avail else f"GPU: {gpu_name} (CUDA 不可用)")
    except Exception as e:
        st.caption(f"GPU: 探测失败 ({e})")
    source_type = st.selectbox(
        "视频源类型",
        list(SOURCE_LABELS.keys()),
        format_func=lambda x: SOURCE_LABELS[x],
    )
    device = st.selectbox("设备", ["cuda:0", "auto", "cpu"], index=0)
    st.caption("显示控制（运行中可实时切换）")

    def _update_display_flag(widget_key, attr_name):
        """Callback: update running system's display attribute on checkbox change."""
        sys_ref = st.session_state.get("_system_ref")
        if sys_ref is not None:
            setattr(sys_ref, attr_name, st.session_state[widget_key])

    show_speed_stats = st.checkbox("速度/统计面板", True, key="show_speed_stats",
                                    on_change=lambda: _update_display_flag("show_speed_stats", "show_player_stats"))
    show_court_trajectory = st.checkbox("球场轨迹分析", True, key="show_court_trajectory",
                                         on_change=lambda: _update_display_flag("show_court_trajectory", "show_court_trajectory"))
    st.divider()
    is_running = st.session_state.get("running", False)
    start_btn = st.button("开始识别", type="primary", disabled=is_running)
    stop_btn = st.button("停止识别", disabled=not is_running)

# Defaults for removed advanced settings
court_update_interval = 8.0
court_update_min_quality = 0.5

uploaded_file = None
browser_url = ""
region = [cfg.screen_region.left, cfg.screen_region.top,
          cfg.screen_region.width, cfg.screen_region.height]

if source_type == "video_file":
    uploaded_file = st.file_uploader(
        "上传本地视频 (mp4/avi/mov, 最大 4GB)",
        type=["mp4", "avi", "mov"],
    )
    if uploaded_file is not None:
        size_mb = uploaded_file.size / (1024 * 1024)
        st.caption(f"已选择: {uploaded_file.name} | 大小: {size_mb:.1f} MB")

elif source_type == "browser_headless":
    browser_url = st.text_input("直播网页地址", placeholder="https://...")
    if browser_url and st.button("测试连通性", key="test_browser"):
        with st.spinner("正在测试..."):
            try:
                src = HeadlessBrowserSource(url=browser_url)
                if src.open():
                    r = src.next_frame()
                    src.close()
                    if r.ok and r.frame is not None and r.frame.std() > 1:
                        st.success(f"连通正常 ({r.frame.shape[1]}x{r.frame.shape[0]})")
                    else:
                        st.warning("已打开但视频可能还在加载")
                else:
                    st.error("无法启动无头浏览器")
            except Exception as e:
                st.error(f"测试失败: {e}")

elif source_type == "screen_capture":
    try:
        sw, sh = _detect_screen_size()
    except Exception:
        sw, sh = 1920, 1080
    st.caption(f"主屏幕: {sw} x {sh}")
    if "scr_preset" not in st.session_state:
        st.session_state["scr_preset"] = "全屏"
    preset = st.selectbox("区域预设", PRESET_NAMES, key="scr_preset_sel")
    st.session_state["scr_preset"] = preset
    if preset != "自定义":
        region = _resolve_preset(preset)
        st.info(f"应用预设: left={region[0]} top={region[1]} {region[2]}x{region[3]}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            region[0] = st.slider("left", 0, sw, region[0], step=10)
            region[1] = st.slider("top", 0, sh, region[1], step=10)
        with col2:
            region[2] = st.slider("width", 100, sw, region[2], step=10)
            region[3] = st.slider("height", 100, sh, region[3], step=10)
    if st.button("预览捕获区域", key="preview_region"):
        try:
            src = _make_screen_source(region)
            if src.open():
                r = src.next_frame()
                src.close()
                if r.ok and r.frame is not None:
                    rgb = cv2.cvtColor(r.frame, cv2.COLOR_BGR2RGB)
                    st.image(rgb, caption=f"预览 {region[2]}x{region[3]}", channels="RGB")
        except Exception as e:
            st.error(f"预览失败: {e}")

for k, v in [("running", False), ("last_metrics", None),
             ("last_run_dir", None), ("current_run_id", None),
             ("_system_ref", None), ("_worker_thread", None),
             # Court annotation state machine
             ("_annotation_phase", "idle"),
             ("_annotation_corners", None),      # auto-detected (1080x720 space)
             ("_manual_corners", None),           # user-confirmed (1080x720 space)
             ("_annotation_frame_path", None),
             ("_annotation_frame_shape", None),
             ("_click_count", 0),
             ("_last_click_unix_time", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Display placeholders (created early so annotation UI can use them) ──
video_col, stats_col = st.columns([3, 1])
status_box = st.container()
frame_ph = video_col.empty()
stats_ph = stats_col.empty()

# Immediately show cached frame to prevent flicker on rerun
_cached_frame = st.session_state.get("_cached_display_frame")
if _cached_frame is not None:
    try:
        frame_ph.image(Image.fromarray(_cached_frame), channels="RGB", use_container_width=True)
    except Exception:
        pass


def _build_source():
    if source_type == "video_file":
        if uploaded_file is None:
            return None, "请上传本地视频文件"
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = UPLOADS_DIR / f"upload_{uploaded_file.name}"
        with open(tmp, "wb") as f:
            while True:
                chunk = uploaded_file.read(10 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return VideoFileSource(str(tmp)), None
    if source_type == "browser_headless":
        if not browser_url:
            return None, "请输入直播网页地址"
        return HeadlessBrowserSource(url=browser_url), None
    if source_type == "screen_capture":
        return _make_screen_source(region), None
    return None, "未知视频源类型"


def _save_results(summary, run_id):
    if summary.get("total_frames", 0) <= 0:
        return (None, None)
    if not run_id:
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = RUNS_DIR / run_id
    try:
        m = RunMetrics(
            run_id=run_id,
            source_type=source_type,
            source_ref=str(region) if source_type == "screen_capture" else
                        (browser_url[:80] if source_type == "browser_headless" else
                         (uploaded_file.name if uploaded_file else "")),
            model_path=cfg.default_model,
            device=device,
            conf=cfg.conf,
            imgsz=cfg.imgsz,
            frame_skip=cfg.frame_skip,
            **summary,
        )
        save_metrics_json(m, run_dir)
    except Exception:
        pass
    try:
        conn = init_db()
        insert_run(conn, m)
        conn.close()
    except Exception:
        pass
    return str(run_dir), run_id


def _run_inference(system, adapter, run_id, save_dir):
    """Start worker thread and return immediately. Display handled by Resume loop."""
    st.session_state["running"] = True
    st.session_state["current_run_id"] = run_id
    st.session_state["last_error"] = None
    st.session_state["_system_ref"] = system
    st.session_state["_run_id"] = run_id
    st.session_state["_save_dir"] = save_dir

    display_q: queue.Queue = queue.Queue(maxsize=4)
    system.display_queue = display_q
    stop_event = threading.Event()
    worker_error: dict = {}

    def worker():
        import datetime as _dt
        import traceback as _tb
        _log_dir = os.path.join(getattr(system, "save_dir", tempfile.gettempdir()), "logs")
        os.makedirs(_log_dir, exist_ok=True)
        _wlog_path = os.path.join(_log_dir, "debug_worker.log")
        try:
            _wlog = open(_wlog_path, "a", encoding="utf-8")
            _wlog.write(f"\n[{_dt.datetime.now()}] worker START\n")
            _wlog.flush()
        except Exception:
            _wlog = None
        try:
            system.process_video(stop_event=stop_event)
            if _wlog:
                _wlog.write(f"[{_dt.datetime.now()}] worker process_video returned OK\n")
                _wlog.flush()
        except Exception as e:
            tb_text = _tb.format_exc()
            worker_error["error"] = repr(e)
            worker_error["traceback"] = tb_text
            print(f"[worker] error: {e}\n{tb_text}")
            if _wlog:
                _wlog.write(f"[{_dt.datetime.now()}] worker EXCEPTION: {e}\n{tb_text}")
                _wlog.flush()
        finally:
            if _wlog:
                try:
                    _wlog.write(f"[{_dt.datetime.now()}] worker FINALLY, closing\n")
                    _wlog.flush()
                    _wlog.close()
                except Exception:
                    pass
            stop_event.set()
            adapter.release()
            try:
                display_q.put_nowait((None, -1))
            except Exception:
                pass

    wt = threading.Thread(target=worker, daemon=True)
    wt.start()
    st.session_state["_worker_thread"] = wt
    st.session_state["_stop_event"] = stop_event
    st.session_state["_worker_error"] = worker_error
    st.session_state["_annotation_phase"] = "idle"

    with status_box:
        st.info(f"识别中... run_id={run_id}（按停止识别中断）")

    st.rerun()


# ── Handle stop button ──
if stop_btn:
    st.session_state["running"] = False
    st.session_state["_annotation_phase"] = "idle"
    stop_evt = st.session_state.get("_stop_event")
    if stop_evt is not None:
        stop_evt.set()
    sys_ref = st.session_state.get("_system_ref")

    # Wait for worker thread to finish
    wt = st.session_state.get("_worker_thread")
    if wt is not None and wt.is_alive():
        wt.join(timeout=3.0)

    if sys_ref is not None:
        try:
            sys_ref.frame_source.release()
        except Exception:
            pass
        # Run post-analysis after stopping
        if hasattr(sys_ref, "get_summary"):
            summary = sys_ref.get_summary()
        else:
            summary = {
                "total_frames": 0, "avg_fps": 0.0, "avg_player_count": 0.0,
                "ball_visible_ratio": 0.0, "upper_avg_speed": 0.0,
                "lower_avg_speed": 0.0, "total_rallies": 0,
            }
        run_id_ref = st.session_state.get("_run_id", "unknown")
        rd, _rid = _save_results(summary, run_id_ref)
        if rd:
            st.session_state["last_metrics"] = summary
            st.session_state["last_run_dir"] = rd
            save_dir_ref = st.session_state.get("_save_dir", "")
            if save_dir_ref:
                detections_path = str(Path(save_dir_ref) / "detections.jsonl")
                try:
                    analysis_result = run_post_analysis(detections_path, str(save_dir_ref))
                    st.session_state["last_analysis"] = analysis_result
                except Exception as e:
                    print(f"[post_analysis] error: {e}")
                    st.session_state["last_analysis"] = None
    # Clear the display so it doesn't look like capture is still running
    st.session_state["_cached_display_frame"] = None
    frame_ph.empty()
    stats_ph.empty()
    with status_box:
        st.info("识别已停止，全场数据与热力图正在生成中，请稍等片刻...")

# ── Resume display loop on rerun (after checkbox toggle, etc.) ──
if st.session_state.get("running", False):
    wt = st.session_state.get("_worker_thread")
    if wt is not None and wt.is_alive():
        # Re-enter the display loop for another 2-second cycle
        display_q_ref = None
        sys_ref = st.session_state.get("_system_ref")
        if sys_ref is not None and sys_ref.display_queue is not None:
            display_q_ref = sys_ref.display_queue
        stop_event_ref = st.session_state.get("_stop_event")

        if display_q_ref is not None and stop_event_ref is not None:
            with status_box:
                st.info("识别中...（按停止识别中断）")

            display_max_w = 960
            display_interval = 1.0 / 10.0
            last_display = 0.0
            last_stats_update = 0.0
            resume_loop_start = time.time()

            while not stop_event_ref.is_set():
                try:
                    item = display_q_ref.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item[0] is None:
                    break

                now = time.time()
                if now - last_display < display_interval:
                    continue
                last_display = now
                frame_bgr, _idx = item

                try:
                    h, w = frame_bgr.shape[:2]
                    if w > display_max_w:
                        scale = display_max_w / w
                        frame_bgr = cv2.resize(frame_bgr, (display_max_w, int(h * scale)),
                                               interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_ph.image(Image.fromarray(rgb), channels="RGB", use_container_width=True)
                    st.session_state["_cached_display_frame"] = rgb
                except Exception:
                    pass

                if now - last_stats_update >= 5.0:
                    last_stats_update = now
                    try:
                        stats_ph.markdown(f"**Frame** {_idx} | **Source** {source_type}")
                    except Exception:
                        pass

            # Worker finished during this rerun cycle
            st.session_state["running"] = False
            frame_ph.empty()
            stats_ph.empty()
            worker_err = st.session_state.get("_worker_error", {})
            err_msg = worker_err.get("error") if isinstance(worker_err, dict) else None
            err_tb = worker_err.get("traceback") if isinstance(worker_err, dict) else None
            with status_box:
                if err_msg:
                    st.error(f"识别异常退出: {err_msg}")
                    if err_tb:
                        with st.expander("查看完整错误栈"):
                            st.code(err_tb, language="text")
                else:
                    st.info("识别已完成")
            run_id_ref = st.session_state.get("_run_id", "unknown")
            sys_ref = st.session_state.get("_system_ref")
            if sys_ref is not None and hasattr(sys_ref, "get_summary"):
                summary = sys_ref.get_summary()
            else:
                summary = {
                    "total_frames": 0, "avg_fps": 0.0, "avg_player_count": 0.0,
                    "ball_visible_ratio": 0.0, "upper_avg_speed": 0.0,
                    "lower_avg_speed": 0.0, "total_rallies": 0,
                }
            rd, _rid = _save_results(summary, run_id_ref)
            if rd:
                st.session_state["last_metrics"] = summary
                st.session_state["last_run_dir"] = rd
                # Post-analysis: generate per-rally heatmaps and scatter plots
                save_dir_ref = st.session_state.get("_save_dir", "")
                if save_dir_ref:
                    detections_path = str(Path(save_dir_ref) / "detections.jsonl")
                    try:
                        analysis_result = run_post_analysis(detections_path, str(save_dir_ref))
                        st.session_state["last_analysis"] = analysis_result
                    except Exception as e:
                        print(f"[post_analysis] error: {e}")
                        st.session_state["last_analysis"] = None
        else:
            st.session_state["running"] = False
    else:
        st.session_state["running"] = False

def _annotation_flow():
    """4-phase state machine: idle → preview → adjust → ready / skip."""
    phase = st.session_state.get("_annotation_phase", "idle")
    if phase == "idle":
        return

    # ── preview: capture frame + auto-detect ──
    if phase == "preview":
        frame_path = st.session_state.get("_annotation_frame_path")
        if frame_path is None:
            try:
                src, err = _build_source()
                if err:
                    st.error(err)
                    st.session_state["_annotation_phase"] = "idle"
                    st.rerun()
                    return
                if not src.open():
                    st.error(f"无法打开视频源: {SOURCE_LABELS[source_type]}")
                    st.session_state["_annotation_phase"] = "idle"
                    st.rerun()
                    return
                result = src.next_frame()
                src.close()
                if not result.ok or result.frame is None:
                    st.error("帧抓取失败")
                    st.session_state["_annotation_phase"] = "idle"
                    st.rerun()
                    return
                frame_path = str(Path(tempfile.gettempdir()) / "court_preview_frame.png")
                cv2.imwrite(frame_path, result.frame)
                st.session_state["_annotation_frame_path"] = frame_path
                st.session_state["_annotation_frame_shape"] = result.frame.shape[:2]
                # Auto-detect on 1080x720
                fixed_w, fixed_h = 1080, 720
                base = cv2.resize(result.frame, (fixed_w, fixed_h))
                auto_corners, _mask, _debug = auto_detect_court_corners(base)
                st.session_state["_annotation_corners"] = auto_corners
                # Seed manual corners with auto-detection result (or None)
                st.session_state["_manual_corners"] = list(auto_corners) if auto_corners else None
                st.session_state["_click_count"] = 4 if auto_corners else 0
            except Exception as e:
                st.error(f"预览失败: {e}")
                st.session_state["_annotation_phase"] = "idle"
                st.rerun()
                return

        img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            st.error("无法读取预览帧")
            st.session_state["_annotation_phase"] = "idle"
            st.rerun()
            return

        corners = st.session_state.get("_annotation_corners")
        img_display = cv2.resize(img_bgr, (1080, 720))

        if corners:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(img_display, [pts], True, (0, 255, 0), 3)
            for idx, pt in enumerate(corners):
                cx, cy = int(pt[0]), int(pt[1])
                cv2.circle(img_display, (cx, cy), 8, (0, 0, 255), -1)
                cv2.putText(img_display, f"{idx+1}", (cx + 12, cy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            with status_box:
                st.info("已自动检测球场边界，建议点击「手动修正」确认角点位置是否准确")
            frame_ph.image(Image.fromarray(rgb), channels="RGB",
                           caption="自动检测结果", use_container_width=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("手动修正角点", type="primary", key="ann_adjust",
                             use_container_width=True):
                    st.session_state["_annotation_phase"] = "adjust"
                    st.session_state["_manual_corners"] = []
                    st.session_state["_last_click_unix_time"] = 0
                    st.rerun()
            with c2:
                if st.button("确认并运行", key="ann_skip"):
                    st.session_state["_annotation_phase"] = "skip"
                    st.rerun()
            with c3:
                if st.button("重新捕获画面", key="ann_recapture"):
                    st.session_state["_annotation_frame_path"] = None
                    st.session_state["_annotation_corners"] = None
                    st.session_state["_manual_corners"] = None
                    st.session_state["_click_count"] = 0
                    st.rerun()
        else:
            rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            with status_box:
                st.warning("未检测到球场边界，请重新捕获或手动标注 4 个角点")
            frame_ph.image(Image.fromarray(rgb), channels="RGB",
                           caption="当前画面", use_container_width=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("重新捕获画面", type="primary", key="ann_recapture2",
                             use_container_width=True):
                    st.session_state["_annotation_frame_path"] = None
                    st.session_state["_annotation_corners"] = None
                    st.session_state["_manual_corners"] = None
                    st.session_state["_click_count"] = 0
                    st.rerun()
            with c2:
                if st.button("手动标注角点", key="ann_adjust2"):
                    st.session_state["_annotation_phase"] = "adjust"
                    st.session_state["_manual_corners"] = []
                    st.session_state["_last_click_unix_time"] = 0
                    st.rerun()
            with c3:
                if st.button("跳过标注直接运行", key="ann_skip2"):
                    st.session_state["_annotation_phase"] = "skip"
                    st.rerun()

    # ── adjust: manual corner annotation with click + number inputs + court preview ──
    elif phase == "adjust":
        frame_path = st.session_state.get("_annotation_frame_path")
        if not frame_path:
            st.session_state["_annotation_phase"] = "preview"
            st.rerun()
            return

        img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            st.error("无法读取预览帧")
            st.session_state["_annotation_phase"] = "idle"
            st.rerun()
            return

        fixed_w, fixed_h = 1080, 720
        img_display = cv2.resize(img_bgr, (fixed_w, fixed_h))

        # Initialize manual corners from auto-detection or empty list
        manual = st.session_state.get("_manual_corners")
        if manual is None:
            manual = []
            st.session_state["_manual_corners"] = manual

        CORNER_LABELS = ["① 左上角", "② 右上角", "③ 右下角", "④ 左下角"]

        # Clear the frame placeholder so old images don't linger
        frame_ph.empty()

        click_count = len(manual)

        # ── Top: prominent instruction banner ──
        if click_count < 4:
            next_label = CORNER_LABELS[click_count]
            st.markdown(
                f"<div style='text-align:center; padding:12px; margin-bottom:8px;"
                f" background:#2a2a2a; border-radius:8px; border:2px solid #FFD700'>"
                f"<span style='font-size:24px; color:#FFD700'>请点击下方图片中的：{next_label}</span>"
                f"<br><span style='font-size:16px; color:#aaa'>进度 {click_count} / 4"
                f"  顺序：左上 → 右上 → 右下 → 左下</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align:center; padding:12px; margin-bottom:8px;"
                " background:#1a3a1a; border-radius:8px; border:2px solid #00E676'>"
                "<span style='font-size:24px; color:#00E676'>✓ 4 个角点标注完成</span>"
                "<br><span style='font-size:16px; color:#aaa'>可微调坐标或确认开始识别</span></div>",
                unsafe_allow_html=True,
            )

        # ── Middle: full-width clickable annotation image (no guides) ──
        rgb_for_click = cv2.cvtColor(img_display.copy(), cv2.COLOR_BGR2RGB)

        # Draw already-marked corners with large red markers
        for idx, pt in enumerate(manual):
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(rgb_for_click, (px, py), 14, (0, 0, 255), -1)
            cv2.putText(rgb_for_click, str(idx + 1),
                        (px + 18, py - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        if len(manual) > 1:
            pts_arr = np.array(manual, dtype=np.int32)
            cv2.polylines(rgb_for_click, [pts_arr], len(manual) == 4,
                          (0, 255, 0), 2)

        # ── Clickable image (FIXED key — critical for reliability) ──
        click_result = streamlit_image_coordinates(
            Image.fromarray(rgb_for_click),
            key="court_click_main",
        )

        if click_result is not None:
            click_unix = click_result.get("unix_time", 0)
            last_unix = st.session_state.get("_last_click_unix_time", 0)
            if click_unix != last_unix and len(manual) < 4:
                st.session_state["_last_click_unix_time"] = click_unix
                cx, cy = click_result["x"], click_result["y"]
                manual.append((cx, cy))
                st.session_state["_manual_corners"] = manual
                st.rerun()

        # ── Action buttons (right after the clickable image) ──
        if len(manual) == 4:
            c1, c2, c3 = st.columns(3)
            is_running = st.session_state.get("running", False)
            is_ready = st.session_state.get("_annotation_phase") == "ready"
            btn_disabled = is_running or is_ready
            with c1:
                if st.button("确认开始识别", type="primary", key="ann_confirm",
                             use_container_width=True, disabled=btn_disabled):
                    # Re-check state to prevent disabled button from executing
                    if not (st.session_state.get("running", False) or 
                            st.session_state.get("_annotation_phase") == "ready"):
                        st.session_state["_annotation_phase"] = "ready"
                        st.rerun()
            with c2:
                if st.button("重新标注（清除所有点）", key="ann_reset_corners", disabled=btn_disabled):
                    if not (st.session_state.get("running", False) or 
                            st.session_state.get("_annotation_phase") == "ready"):
                        st.session_state["_manual_corners"] = []
                        st.session_state["_click_count"] = 0
                        st.session_state["_last_click_unix_time"] = 0
                        st.rerun()
            with c3:
                if st.button("返回重新捕获画面", key="ann_back_to_preview", disabled=btn_disabled):
                    if not (st.session_state.get("running", False) or 
                            st.session_state.get("_annotation_phase") == "ready"):
                        st.session_state["_annotation_frame_path"] = None
                        st.session_state["_annotation_corners"] = None
                        st.session_state["_manual_corners"] = None
                        st.session_state["_click_count"] = 0
                        st.session_state["_last_click_unix_time"] = 0
                        st.session_state["_annotation_phase"] = "preview"
                        st.rerun()
        else:
            c1, c2 = st.columns(2)
            is_running = st.session_state.get("running", False)
            is_ready = st.session_state.get("_annotation_phase") == "ready"
            btn_disabled = is_running or is_ready
            with c1:
                if st.button("重新标注（清除所有点）", key="ann_reset_corners2", disabled=btn_disabled):
                    if not (st.session_state.get("running", False) or 
                            st.session_state.get("_annotation_phase") == "ready"):
                        st.session_state["_manual_corners"] = []
                        st.session_state["_click_count"] = 0
                        st.session_state["_last_click_unix_time"] = 0
                        st.rerun()
            with c2:
                if st.button("返回重新捕获画面", key="ann_back_to_preview2", disabled=btn_disabled):
                    if not (st.session_state.get("running", False) or 
                            st.session_state.get("_annotation_phase") == "ready"):
                        st.session_state["_annotation_frame_path"] = None
                        st.session_state["_annotation_corners"] = None
                        st.session_state["_manual_corners"] = None
                        st.session_state["_click_count"] = 0
                        st.session_state["_last_click_unix_time"] = 0
                        st.session_state["_annotation_phase"] = "preview"
                        st.rerun()

        # ── Collapsible sections (reference image + court preview) ──
        with st.expander("查看原始捕获画面（参考）", expanded=False):
            ref_rgb = cv2.cvtColor(img_display.copy(), cv2.COLOR_BGR2RGB)
            st.image(Image.fromarray(ref_rgb), caption="屏幕捕获的原始画面 — 仅供对照参考",
                     use_container_width=True)

        # ── Number inputs + court preview (only after 4 corners) ──
        if len(manual) == 4:
            with st.expander("微调角点坐标 + 球场线预览", expanded=False):
                st.markdown("修改坐标后自动刷新预览")
                cols = st.columns(4)
                changed = False
                for i, (label, col) in enumerate(zip(CORNER_LABELS, cols)):
                    with col:
                        new_x = st.number_input(
                            f"{label} X", 0, fixed_w - 1, int(manual[i][0]),
                            key=f"corner_x_{i}", step=5,
                        )
                        new_y = st.number_input(
                            f"{label} Y", 0, fixed_h - 1, int(manual[i][1]),
                            key=f"corner_y_{i}", step=5,
                        )
                        if (new_x, new_y) != (int(manual[i][0]), int(manual[i][1])):
                            manual[i] = (new_x, new_y)
                            changed = True
                if changed:
                    st.session_state["_manual_corners"] = manual
                    st.rerun()

                # ── Court lines preview using CourtMapper ──
                preview_img = img_display.copy()
                try:
                    mapper = CourtMapper(manual)
                    overlay, mid_h = mapper.draw_court_overlay(preview_img)
                    preview_img = overlay
                except Exception:
                    cv2.putText(preview_img, "Court overlay error", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                for idx, pt in enumerate(manual):
                    cv2.circle(preview_img, tuple(int(c) for c in pt), 14, (0, 0, 255), -1)
                    cv2.putText(preview_img, str(idx + 1),
                                (int(pt[0]) + 18, int(pt[1]) - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

                rgb_preview = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
                st.image(Image.fromarray(rgb_preview), channels="RGB",
                         caption="球场线叠加预览（绿线应与画面中球场边界重合）",
                         use_container_width=True)

    # ── ready: use confirmed manual corners + start inference ──
    elif phase == "ready":
        # Clear previous analysis results to avoid showing stale data during new run
        st.session_state["last_analysis"] = None
        
        corners = st.session_state.get("_manual_corners")
        if not corners or len(corners) != 4:
            st.error("角点数据不完整，请返回重新标注")
            st.session_state["_annotation_phase"] = "adjust"
            st.rerun()
            return

        frame_path = st.session_state["_annotation_frame_path"]
        orig_h, orig_w = st.session_state["_annotation_frame_shape"]
        fixed_w, fixed_h = 1080, 720
        sx, sy = orig_w / fixed_w, orig_h / fixed_h
        orig_corners = [(int(x * sx), int(y * sy)) for x, y in corners]
        mapper = CourtMapper(orig_corners)
        mid_height = mapper.mid_height

        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        save_dir = RUNS_DIR / run_id
        save_dir.mkdir(parents=True, exist_ok=True)
        roi_corners = compute_expanded_roi(orig_corners, (orig_h, orig_w))
        with open(str(save_dir / "court_annotations.txt"), 'w') as f:
            f.write(f"corners={orig_corners}\n")
            f.write(f"roi_corners={roi_corners}\n")
            f.write(f"mid_height={mid_height}\n")
            f.write(f"source=manual\n")
        shutil.copy2(frame_path, str(save_dir / "first_frame.png"))

        src, err = _build_source()
        if err:
            st.error(err); st.session_state["_annotation_phase"] = "idle"; st.rerun(); return
        if not src.open():
            st.error("无法打开视频源"); st.session_state["_annotation_phase"] = "idle"; st.rerun(); return
        first = src.next_frame()
        if not first.ok or first.frame is None:
            src.close(); st.error("帧抓取失败"); st.session_state["_annotation_phase"] = "idle"; st.rerun(); return
        adapter = StreamAdapter(source=src, fps=30.0, first_frame=first.frame)
        is_realtime = source_type in ("screen_capture", "browser_headless")
        system = BadmintonAnalysisSystem(
            video_path=f"{run_id}.mp4", template_path=str(save_dir / "first_frame.png"),
            output_dir=str(save_dir), ball_model_path="weights/yolo11s-ball.pt",
            pose_family="yolo-pose", yolo_pose_model="weights/yolo11n-pose.pt",
            show_display=False, show_skeletons=True, show_player_trajectories=True,
            show_court_trajectory=show_court_trajectory, show_shuttlecock_trajectory=True,
            show_player_stats=show_speed_stats, frame_source=adapter,
            non_interactive_annotation=True, skip_court_annotation=False,
            device=device if device != "auto" else None,
            court_update_interval=court_update_interval,
            court_update_min_quality=court_update_min_quality,
            enable_court_updater=False,
            save_video=not is_realtime,
        )
        system.keep_audio = False
        if not _run_inference(system, adapter, run_id, save_dir):
            st.rerun()
            return
        st.session_state["_annotation_phase"] = "idle"

    # ── skip: use auto-detected corners (or full-frame fallback) ──
    elif phase == "skip":
        frame_path = st.session_state.get("_annotation_frame_path")
        auto_corners = st.session_state.get("_annotation_corners")
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        save_dir = RUNS_DIR / run_id
        save_dir.mkdir(parents=True, exist_ok=True)
        src, err = _build_source()
        if err:
            st.error(err); st.session_state["_annotation_phase"] = "idle"; st.rerun(); return
        if not src.open():
            st.error("无法打开视频源"); st.session_state["_annotation_phase"] = "idle"; st.rerun(); return
        first = src.next_frame()
        if not first.ok or first.frame is None:
            src.close(); st.error("帧抓取失败"); st.session_state["_annotation_phase"] = "idle"; st.rerun(); return

        # Write court annotations using auto-detected corners (scaled to original resolution)
        # or fall back to full-frame defaults if auto-detection failed
        orig_h, orig_w = first.frame.shape[:2]
        if auto_corners and len(auto_corners) == 4:
            fixed_w, fixed_h = 1080, 720
            sx, sy = orig_w / fixed_w, orig_h / fixed_h
            orig_corners = [(int(x * sx), int(y * sy)) for x, y in auto_corners]
            mapper = CourtMapper(orig_corners)
            mid_height = mapper.mid_height
            roi_corners = compute_expanded_roi(orig_corners, (orig_h, orig_w))
            with open(str(save_dir / "court_annotations.txt"), 'w') as f:
                f.write(f"corners={orig_corners}\n")
                f.write(f"roi_corners={roi_corners}\n")
                f.write(f"mid_height={mid_height}\n")
                f.write(f"source=auto_skip\n")
            skip_court = False
        else:
            # No auto-detected corners: fall back to full-frame defaults
            skip_court = True

        first_path = save_dir / "first_frame.png"
        cv2.imwrite(str(first_path), first.frame)
        adapter = StreamAdapter(source=src, fps=30.0, first_frame=first.frame)
        is_realtime = source_type in ("screen_capture", "browser_headless")
        system = BadmintonAnalysisSystem(
            video_path=f"{run_id}.mp4", template_path=str(first_path),
            output_dir=str(save_dir), ball_model_path="weights/yolo11s-ball.pt",
            pose_family="yolo-pose", yolo_pose_model="weights/yolo11n-pose.pt",
            show_display=False, show_skeletons=True, show_player_trajectories=True,
            show_court_trajectory=show_court_trajectory, show_shuttlecock_trajectory=True,
            show_player_stats=show_speed_stats, frame_source=adapter,
            non_interactive_annotation=True, skip_court_annotation=skip_court,
            device=device if device != "auto" else None,
            court_update_interval=court_update_interval,
            court_update_min_quality=court_update_min_quality,
            enable_court_updater=False,
            save_video=not is_realtime,
        )
        system.keep_audio = False
        if not _run_inference(system, adapter, run_id, save_dir):
            st.rerun()
            return
        st.session_state["_annotation_phase"] = "idle"


# ── Start annotation flow on button press ──
if start_btn and not st.session_state["running"]:
    if st.session_state["_annotation_phase"] == "idle":
        st.session_state["_annotation_phase"] = "preview"
        st.session_state["_annotation_frame_path"] = None
        st.session_state["_annotation_corners"] = None
        st.session_state["_annotation_frame_shape"] = None

# Run the annotation state machine
_annotation_flow()

# ── Post-analysis results display ──
_analysis = st.session_state.get("last_analysis")
if _analysis and _analysis.get("heatmap_paths"):
    st.divider()
    st.subheader(f"后期分析结果（共 {_analysis.get('num_rallies', 0)} 个回合）")
    save_dir_ref = st.session_state.get("_save_dir", "")
    if save_dir_ref:
        st.markdown(f"结果保存路径: `{save_dir_ref}`")

    # Match-wide heatmap + scatter side by side
    heatmaps = _analysis.get("heatmap_paths", [])
    scatters = _analysis.get("scatter_paths", [])
    match_hm = [p for p in heatmaps if "match_heatmap" in p]
    match_sc = [p for p in scatters if "match_scatter" in p]
    if match_hm or match_sc:
        st.markdown("**全场汇总**")
        mc1, mc2 = st.columns(2)
        if match_hm:
            with mc1:
                st.image(match_hm[0], caption="全场热力图", use_container_width=True)
        if match_sc:
            with mc2:
                st.image(match_sc[0], caption="全场散点图", use_container_width=True)

    # Per-rally images in expanders
    rally_hms = [p for p in heatmaps if "rally_" in p]
    rally_scs = [p for p in scatters if "rally_" in p]
    if rally_hms or rally_scs:
        with st.expander("各回合详情", expanded=False):
            rally_ids = sorted(set(
                int(p.split("rally_")[1].split("_")[0]) for p in rally_hms + rally_scs
            ))
            for rid in rally_ids:
                st.markdown(f"**回合 {rid}**")
                rc1, rc2 = st.columns(2)
                hm = [p for p in rally_hms if f"rally_{rid}_" in p]
                sc = [p for p in rally_scs if f"rally_{rid}_" in p]
                if hm:
                    with rc1:
                        st.image(hm[0], caption=f"回合 {rid} 热力图", use_container_width=True)
                if sc:
                    with rc2:
                        st.image(sc[0], caption=f"回合 {rid} 散点图", use_container_width=True)

    # Movement stats table
    rally_stats = _analysis.get("rally_stats", {})
    if rally_stats:
        st.markdown("**球员运动统计**")
        rows = []
        for rid in sorted(rally_stats.keys()):
            for half, label in [("upper", "上场"), ("lower", "下场")]:
                if half in rally_stats[rid]:
                    s = rally_stats[rid][half]
                    rows.append({
                        "回合": rid, "半场": label,
                        "均速(m/s)": s.get("avg_speed", 0),
                        "极速(m/s)": s.get("max_speed", 0),
                        "距离(m)": s.get("total_distance", 0),
                    })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


st.divider()
st.subheader("数据文件位置（均在项目文件夹内）")

st.markdown(
    f"""
| 目录 | 路径 | 内容 |
|---|---|---|
| 上传视频 | `{UPLOADS_DIR}` | Web 上传的本地视频 |
| 模型权重 | `{Path('models')}` | yolo11n-pose.pt 等 |
| 识别结果 | `{RUNS_DIR}` | 每轮次的 metrics.json + 标注视频 |
| SQLite 数据库 | `{OUTPUTS_DIR / 'badminton.db'}` | 历史识别记录 |
"""
)
