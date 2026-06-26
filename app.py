"""Streamlit Web UI for Good-Badminton.

Mirrors the structure of football-realtime-analyzer/app.py
(sidebar + main + history) but uses badminton-native detection
(YOLO Pose + ball detection + court mapping + upper/lower player tracking).
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from badminton_analysis.config import (
    OUTPUTS_DIR,
    RUNS_DIR,
    UPLOADS_DIR,
    VIDEOS_DIR,
    load_config,
)
from badminton_analysis.renderer import render_frame
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
    list_runs,
    save_metrics_json,
)
from badminton_analysis.system import BadmintonAnalysisSystem, load_runtime_dependencies

# Eagerly load heavy runtime dependencies (YOLO model, etc.) once at app start,
# so that subsequent "开始识别" clicks don't take 5-10s to spin up the model.
with st.spinner("加载 YOLO 模型中（约 5-10 秒）..."):
    load_runtime_dependencies()

st.set_page_config(page_title="Good-Badminton 实时识别系统", layout="wide")
st.title("基于 YOLO 的羽毛球比赛多源实时目标检测与分析系统")

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
    source_type = st.selectbox(
        "视频源类型",
        list(SOURCE_LABELS.keys()),
        format_func=lambda x: SOURCE_LABELS[x],
    )
    device = st.selectbox("设备", ["cuda:0", "auto", "cpu"], index=0)
    model_path = st.text_input("模型路径", cfg.default_model)
    imgsz = st.slider("imgsz (推理分辨率)", 320, 1280, cfg.imgsz, step=64)
    conf = st.slider("置信度阈值 conf", 0.05, 0.9, cfg.conf, step=0.05)
    frame_skip = st.slider("帧跳过 frame_skip", 1, 10, cfg.frame_skip, step=1)
    no_court = st.checkbox("跳过球场自动检测 (--no-court)", False)
    show_raw = st.checkbox("显示原始画面（调试用）", False)
    save_output = st.checkbox("保存标注视频", cfg.save_output)
    st.divider()
    start_btn = st.button("开始识别", type="primary")
    stop_btn = st.button("停止识别")

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
             ("last_run_dir", None), ("current_run_id", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

if stop_btn:
    st.session_state["running"] = False


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
        return None
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
            model_path=model_path,
            device=device,
            conf=conf,
            imgsz=imgsz,
            frame_skip=frame_skip,
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


video_col, stats_col = st.columns([3, 1])
status_box = st.container()
frame_ph = video_col.empty()
stats_ph = stats_col.empty()


def _run_inference_thread(system, adapter, run_id, save_dir):
    """Spawn a background thread that drives process_video and pushes frames."""
    st.session_state["running"] = True
    st.session_state["current_run_id"] = run_id
    display_q: queue.Queue = queue.Queue(maxsize=2)
    system.display_queue = display_q
    stop_event = threading.Event()

    def worker():
        try:
            system.process_video()
        except Exception as e:
            print(f"[worker] error: {e}")
        finally:
            stop_event.set()
            adapter.release()
            try:
                display_q.put_nowait((None, -1))  # None is the END sentinel
            except Exception:
                pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    with status_box:
        st.info(f"识别中... run_id={run_id}")

    last_metrics = {"total_frames": 0, "avg_fps": 0.0, "total_rallies": 0}
    while st.session_state.get("running", False) and not stop_event.is_set():
        try:
            item = display_q.get(timeout=0.3)
        except queue.Empty:
            continue
        if item[0] is None:  # END sentinel (frame items are np.ndarray, never None)
            break
        frame_bgr, _idx = item
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_ph.image(Image.fromarray(rgb), channels="RGB", use_container_width=True)
        last_metrics["total_frames"] = _idx + 1
        stats_ph.markdown(
            f"**FPS** (实时)\n\n**Frame** {_idx}\n\n"
            f"**Status** running\n\n**Source** {source_type}"
        )

    t.join(timeout=5)
    st.session_state["running"] = False
    summary = {
        "total_frames": last_metrics["total_frames"],
        "avg_fps": 0.0, "avg_player_count": 0.0,
        "ball_visible_count": 0, "ball_visible_ratio": 0.0,
        "upper_player_count": 0, "lower_player_count": 0,
        "upper_avg_speed": 0.0, "lower_avg_speed": 0.0,
        "upper_max_speed": 0.0, "lower_max_speed": 0.0,
        "upper_total_distance": 0.0, "lower_total_distance": 0.0,
        "total_rallies": 0,
    }
    rd, _rid = _save_results(summary, run_id)
    if rd:
        st.session_state["last_metrics"] = summary
        st.session_state["last_run_dir"] = rd


if start_btn and not st.session_state["running"]:
    src, err = _build_source()
    if err:
        with status_box:
            st.error(err)
    else:
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        save_dir = RUNS_DIR / run_id
        save_dir.mkdir(parents=True, exist_ok=True)
        if not src.open():
            with status_box:
                st.error(f"无法打开视频源: {SOURCE_LABELS[source_type]}")
        else:
            first = src.next_frame()
            if not first.ok or first.frame is None:
                src.close()
                with status_box:
                    st.error("首帧抓取失败")
            else:
                first_path = save_dir / "first_frame.png"
                cv2.imwrite(str(first_path), first.frame)
                adapter = StreamAdapter(source=src, fps=30.0, first_frame=first.frame)
                system = BadmintonAnalysisSystem(
                    video_path=f"{run_id}.mp4",
                    template_path=str(first_path),
                    output_dir=str(save_dir),
                    ball_model_path="weights/yolo11s-ball.pt",
                    pose_family="yolo-pose",
                    yolo_pose_model="weights/yolo11n-pose.pt",
                    show_display=False,
                    show_skeletons=True,
                    show_player_trajectories=True,
                    show_court_trajectory=not no_court,
                    show_shuttlecock_trajectory=True,
                    show_player_stats=True,
                    frame_source=adapter,
                    non_interactive_annotation=True,
                    skip_court_annotation=no_court,
                )
                system.keep_audio = False
                _run_inference_thread(system, adapter, run_id, save_dir)


st.divider()
st.subheader("本次识别结果")
m = st.session_state.get("last_metrics")
if m:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总帧数", m["total_frames"])
    c2.metric("回合数", m["total_rallies"])
    c3.metric("平均球员数", f"{m['avg_player_count']:.1f}")
    c4.metric("羽毛球可见率", f"{m['ball_visible_ratio']*100:.1f}%")
    p1, p2 = st.columns(2)
    p1.metric("上半场 平均速度", f"{m['upper_avg_speed']:.2f} m/s")
    p2.metric("下半场 平均速度", f"{m['lower_avg_speed']:.2f} m/s")
    if st.session_state.get("last_run_dir"):
        st.code(f"结果目录: {st.session_state['last_run_dir']}")
else:
    st.info("尚未进行识别。点击「开始识别」后这里会显示本次汇总指标。")

st.subheader("历史识别记录 (SQLite)")
try:
    conn = init_db()
    rows = list_runs(conn, limit=20)
    conn.close()
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        show_cols = [
            "run_id", "source_type", "source_ref", "device", "total_frames",
            "avg_fps", "avg_player_count", "ball_visible_ratio",
            "upper_avg_speed", "lower_avg_speed", "total_rallies", "created_at",
        ]
        avail = [c for c in show_cols if c in df.columns]
        st.dataframe(df[avail], use_container_width=True, hide_index=True)
        st.caption(f"数据库位置: {OUTPUTS_DIR / 'football.db'}")
    else:
        st.info("暂无历史记录。")
except Exception as e:
    st.warning(f"读取历史记录失败: {e}")

st.divider()
st.subheader("数据文件位置（均在项目文件夹内）")
st.markdown(
    f"""
| 目录 | 路径 | 内容 |
|---|---|---|
| 上传视频 | `{UPLOADS_DIR}` | Web 上传的本地视频 |
| 模型权重 | `{Path('models')}` | yolo11n-pose.pt 等 |
| 识别结果 | `{RUNS_DIR}` | 每轮次的 metrics.json + 标注视频 |
| SQLite 数据库 | `{OUTPUTS_DIR / 'football.db'}` | 历史识别记录 |
"""
)
