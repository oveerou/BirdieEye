# BirdieEye: AI Badminton Hawk-Eye System 🏸

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/yo-WASSUP/BirdieEye?style=social)](https://github.com/yo-WASSUP/BirdieEye/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yo-WASSUP/BirdieEye?style=social)](https://github.com/yo-WASSUP/BirdieEye/network/members)
[![GitHub license](https://img.shields.io/github/license/yo-WASSUP/BirdieEye)](https://github.com/yo-WASSUP/BirdieEye/blob/main/LICENSE)
[![RedNote](https://img.shields.io/badge/RedNote-ff2442)](https://www.xiaohongshu.com/explore/6a37b1d20000000011016229?xsec_token=ABod3wXBTiDppp6W2Ou0QHlu2eotUkeu27-ha64nFRR74=&xsec_source=pc_user)

**A computer-vision toolkit for badminton match video analysis**

[中文](README.md) | [English](README_EN.md)

</div>

## 🎬 Preview

| Feature | Screenshot |
|---------|-----------|
| **Screen Capture** — capture any screen region for real-time analysis | ![screen capture](screenshots/demo-1.png) |
| **Court Annotation** — mark 4 court corners for coordinate mapping | ![corner annotation](screenshots/demo-2.png) |
| **Live Analysis** — skeleton, trajectory tracking, stats overlay | ![live analysis](screenshots/demo-3.png) |
| **Post-analysis** — per-rally heatmap, scatter plot, movement stats | ![post analysis](screenshots/demo-4.png) |

## 🆕 Changelog

- **2026-06-29**: Added post-analysis module (rally segmentation, KDE heatmaps, scatter plots, movement statistics); court drift correction; GPU FP16/TF32 inference optimization; Web UI interaction improvements (error traceback display, court detection retry, startup reminder).
- **2026-06-27**: Added multi-source real-time input (screen capture / headless browser); Streamlit web console; court model auto-update + real-time heatmap; SQLite history tracking.
- **2026-06-23**: Added automatic court boundary detection.
- **2026-06-20**: Initial open-source release.
- **2026-06-17**: Project documentation cleanup.
- **Current version**: Supports player pose detection, shuttlecock detection, court coordinate mapping, trajectory statistics, heatmaps, scatter plots, and annotated video output.
- **Experimental features**: Hit-point analysis and stroke statistics are still under active iteration and are mainly intended for research and secondary development.

## 🔮 Roadmap

- [x] Frame-by-frame badminton match video analysis
- [x] RTMPose / RTMO / YOLO Pose model support
- [x] YOLO shuttlecock detection model integration
- [x] Manual court annotation and court coordinate mapping
- [x] Player movement trajectory, speed, distance, and rally statistics
- [x] Chinese / English visualization text
- [x] Heatmap, scatter plot, and detection data export
- [x] Automatic court keypoint detection
- [x] Multi-source real-time input (screen capture / headless browser / local video)
- [x] Streamlit web console (one-click start.bat)
- [x] Court model auto-update + drift correction
- [x] Real-time heatmap overlay
- [x] Post-analysis (rally segmentation, KDE heatmaps, scatter plots)
- [x] GPU FP16 / TF32 inference optimization
- [x] SQLite history and metrics storage
- [ ] More stable hit-point recognition
- [ ] More accurate shuttlecock detection model
- [ ] More complete stroke statistics
- [ ] Batch video analysis workflow

---

## ✨ Features

- **Player pose detection** - Supports RTMPose, RTMO, and Ultralytics YOLO Pose for human keypoint and skeleton detection.
- **Shuttlecock detection** - Uses a YOLO model to detect shuttlecock positions and draw trajectories in the output video.
- **Court coordinate mapping** - Manually annotates court keypoints and maps image coordinates to standard badminton court coordinates.
- **Player position tracking** - Tracks upper-court and lower-court players separately and records movement trajectories.
- **Rally detection** - Detects rally start/end states from continuous court-view matching and records rally IDs in overlays and detection data.
- **Motion statistics** - Computes movement distance, current speed, maximum speed, and rally counts.
- **Visual output** - Generates annotated videos with skeletons, trajectories, statistics, and court trajectory overlays.
- **Position charts** - Automatically generates player position heatmaps and scatter plots.
- **Chinese / English display** - Switch visualization text with `--language zh/en`.
- **Local execution** - Videos, models, and analysis outputs stay on your local machine.
- **Multi-source real-time input** - Supports local video, screen capture (mss), and headless browser (Chrome DevTools Protocol) for analyzing live feeds.
- **Court drift correction** - Periodically re-detects court corners via homography to correct camera drift during long sessions, maintaining coordinate mapping accuracy.
- **Real-time heatmap** - 2-minute sliding-window heatmap with separate upper/lower court halves, overlaid as a minimap in the bottom-right corner.
- **Post-analysis** - Automatic rally segmentation, KDE heatmaps, scatter plots, and per-rally movement statistics for publication-quality analysis.
- **GPU inference optimization** - FP16 half-precision inference + TF32 matrix multiplication + cuDNN benchmark for significantly improved GPU frame rates.
- **Web console** - Streamlit interface with real-time frame display, parameter tuning, court annotation, history browsing, and one-click `start.bat` launch.
- **SQLite history** - Each run automatically records metrics to a SQLite database for historical review and comparison.

## Requirements

- Python 3.8+
- Font: Chinese text rendering requires a bold font file (e.g., `simhei.ttf`). Download it from [GitHub Releases](https://github.com/yo-WASSUP/BirdieEye/releases/latest) and place in the project root.
- FFmpeg available in system `PATH`
- Shuttlecock YOLO detection weight, downloaded from [GitHub Releases](https://github.com/yo-WASSUP/BirdieEye/releases/latest)

## Performance Requirements and Reference Speed

Recommended setup:

- GPU with 6GB+ VRAM. More VRAM helps with higher-resolution videos and larger pose models.
- 16GB+ system RAM.
- SSD storage for output videos, `detections.jsonl`, and visualization images.
- CPU execution is supported, but pose detection and shuttlecock detection will be much slower. It is best suited for short clips or feature checks.

Actual speed depends on the GPU, video resolution, pose model, preview display, and audio export settings.

For a 720p video with `--pose-family yolo-pose --yolo-pose-model yolo11n-pose.pt` and `weights/yolo11s-ball.pt`, GPU timing logs are typically close to:

```text
pose 0.02s, shuttlecock 0.02s, shuttle draw 0.00s, players draw 0.01s, court draw 0.00s
```

Use `--performance-stats` to print a compact timing summary about every 5 seconds and identify whether the bottleneck is pose inference, shuttlecock detection, or drawing.

## 🚀 Installation

The default dependencies use CPU PyTorch and ONNX Runtime.

### Windows

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### GPU Acceleration (Windows / NVIDIA)

Prerequisites:

- NVIDIA driver installed, and `nvidia-smi` works correctly.
- CUDA 12.1 PyTorch wheels are recommended.

PowerShell:

```bash
.\.venv\Scripts\activate

pip uninstall -y torch torchvision onnxruntime onnxruntime-gpu
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu==1.20.1
```

Verify GPU availability:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Expected output includes:

```text
cuda: True
CUDAExecutionProvider
```

> Note: after installing GPU ONNX Runtime, `pip check` may report `rtmlib requires onnxruntime, which is not installed`. If provider verification shows `CUDAExecutionProvider`, do not reinstall CPU `onnxruntime`, because it may overwrite the GPU package.

Switch back to CPU dependencies:

```bash
pip install --force-reinstall -r requirements.txt
```


## 📝 Usage


### First Run Workflow

1. Prepare the input video and shuttlecock detection weight.
2. Run the basic command:

```bash
python main.py --video-path videos/demo.mp4
```

3. If `--template-path` is not provided, the program opens a file picker for a court template image. Usually, choose a stable frame with clear court lines.
4. The program first tries to detect the court boundary automatically and saves `outputs/<video_name>/auto_court_preview.png`. Press Enter/Y in the preview window to accept it, or press M/R/Esc to switch to manual four-corner annotation.
5. If manual annotation is used, follow the prompt at the top of the image and click the four court corners in order.

6. After the four points are selected, the window shows a green court box and a blue pose-detection ROI. The ROI is generated automatically from the court area.
7. The annotation is saved to `outputs/<video_name>/court_annotations.txt`. Re-running with the same output directory reuses this file.
8. After analysis finishes, check `outputs/<video_name>/detect_<video_name>.mp4`, `detections.jsonl`, and `position_visualizations/`.

Why four court points are required:

- The four corners establish the mapping from image coordinates to standard badminton court coordinates.
- Player filtering mainly depends on court coordinates, which helps remove spectators, referees, and people outside the court.
- Upper/lower court player assignment, movement distance, speed, rally statistics, heatmaps, and scatter plots all depend on this mapping.
- Rally detection uses court template matching: consecutive court-view frames start a rally, and consecutive non-court-view frames end it.
- The pose ROI only reduces the inference area and improves speed. It is automatically expanded from the court area.
- Shuttlecock detection still runs on the full frame, with basic filtering based on the horizontal court range plus padding.

If the video angle, crop, or template image changes, delete the corresponding `court_annotations.txt` and annotate the four points again.


### Pose Model Selection

```bash
# Default: two-stage RTMPose balanced
python main.py --video-path videos/demo.mp4 --pose-family rtmpose --pose-mode balanced

# Lighter one-stage RTMO
python main.py --video-path videos/demo.mp4 --pose-family rtmo --pose-mode lightweight

# Use Ultralytics YOLO Pose
python main.py --video-path videos/demo.mp4 --pose-family yolo-pose --yolo-pose-model yolo11n-pose.pt
```

RTMPose / RTMO modes:

- `lightweight`: prioritizes speed.
- `balanced`: default tradeoff between speed and quality.
- `performance`: larger model, slower, usually better for detection quality.

### Common Arguments

```text
--video-path                 Input video path, required
--output-dir                 Output directory, default outputs/<video_name>
--ball-model                 YOLO shuttlecock detection model path, default weights/yolo11s-ball.pt
--pose-family                Pose model family: rtmpose, rtmo, or yolo-pose
--pose-mode                  RTMPose / RTMO mode: lightweight, balanced, performance
--yolo-pose-model            YOLO pose model path or model name, default yolo11n-pose.pt
--template-path              Court template image path; opens a file picker if omitted
--pose-roi true|false                Show pose-detection ROI box, default true
--display true|false                 Show OpenCV preview window, default true
--skeletons true|false               Show human skeletons, default true
--player-trajectories true|false     Show player trajectories, default true
--court-trajectory true|false        Show court trajectory overlay, default true
--shuttlecock-trajectory true|false  Show shuttlecock trajectory, default true
--player-stats true|false            Show player statistics, default true
--performance-stats                  Print performance timings
--save-images                        Save processed frame images
--visualize-positions true|false     Generate heatmaps and scatter plots, default true
--audio true|false                   Keep original video audio, default true
--language {zh,en}                   Visualization language
```

## 📊 Outputs

Default output directory: `outputs/<video_name>/`.

- `metadata.json`: metadata for video, models, court annotation, and output files.
- `detections.jsonl`: per-frame detection records, including rally ID, players, hands, court coordinates, speed, and shuttlecock coordinates.
- `detect_<video_name>.mp4`: annotated output video with skeletons, trajectories, statistics, and rally IDs.
- `court_annotations.txt`: cached court annotation coordinates.
- `position_visualizations/heatmaps/`: player position heatmaps.
- `position_visualizations/scatter_plots/`: player position scatter plots.

## 🧩 Project Structure

```text
main.py              # CLI entry for local video analysis
app.py               # Streamlit web console entry
start.bat            # One-click launcher (creates venv + installs deps + starts Streamlit)
badminton_analysis/
├── system.py        # Main video analysis pipeline: BadmintonAnalysisSystem
├── config.py        # YAML config loader and AppConfig
├── storage.py       # SQLite history + metrics.json writer
├── analytics/       # Real-time heatmap + post-analysis (rally segmentation, KDE, scatter)
├── sources/         # Multi-source input (screen capture / headless browser / video file / stream adapter)
├── court/           # Court detection, coordinate mapping, drift correction, auto-update
├── detection/       # Shuttlecock detection and pose detection
├── tracking/        # Player tracking
├── visualization/   # Video overlays, statistics charts, and position plots
└── media/           # Video/audio processing
```

## Acknowledgements

Thanks to the RTMPose, RTMO, and OpenMMLab ecosystem for the pose-estimation foundations, and to [Tau-J/rtmlib](https://github.com/Tau-J/rtmlib) for the lightweight pose-estimation runtime.

Thanks to [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO object-detection algorithms and tooling.

Thanks to [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) for organizing and releasing badminton datasets, which provided important references for shuttlecock detection and trajectory analysis in this project.

## 📄 License

Project code and `weights/yolo11s-ball.pt` are licensed under Apache License 2.0. RTMPose / RTMO / YOLOX ONNX weights provided in Releases come from the OpenMMLab / RTMPose ecosystem, are used under their upstream Apache License 2.0, and retain their original attribution.

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


## Web console (Streamlit)

In addition to the CLI (`main.py` / `live.py`), a Streamlit web console is also provided. The structure mirrors `football-realtime-analyzer` (sidebar + main + history), with badminton-native detection (YOLO Pose + ball detection + court mapping + upper/lower player tracking).

### Launch

```bat
:: One-click launch (creates venv + installs deps + starts Streamlit on first run)
start.bat

:: Or manual
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501
```

Open http://localhost:8501 in your browser.

### Video sources

The sidebar lets you pick one of three:
- **Local video**: upload mp4/avi/mov (max 4 GB)
- **Web live stream**: paste an https URL; headless Chrome grabs the `<video>` element
- **Screen capture**: pick a preset (Fullscreen / Left half / Right half / Top half / Bottom half / Center 1280x720) or define a custom rectangle; a "Preview region" button is provided.

### Real-time stats

The right column refreshes each frame with:
- FPS / frame index / device
- Detected player count / shuttlecock visibility
- Current rally id
- Upper-court + lower-court player count / average speed / cumulative distance

After the run, results are written to `outputs/badminton.db` (SQLite) and the bottom table shows the history.

### Court detection options

- Default: auto-detect (requires a clear court in the frame) + non-interactive mode
- "Skip court auto-detection" checkbox: enables --no-court mode, which works on any content (no court mapping)

### File locations

| Directory | Contents |
|---|---|
| `outputs/runs/run_<timestamp>_<id>/` | per-run `metrics.json` + annotated video |
| `outputs/badminton.db` | SQLite history |
| `outputs/uploads/` | videos uploaded through the web UI |

### Known limits

- Screen source is non-interactive (no OpenCV popup), so "skip court auto-detection" is the recommended option for screen capture.
- Streamlit is single-user; multi-user needs an auth layer.
- For long videos, raise `frame_skip` to save GPU.


## Court model auto-update + real-time heatmap

During playback, "full-court" frames (two players clearly visible + sharp image) are used to automatically refresh the court model. A 2-minute sliding-window heatmap (upper/lower halves shown separately) is overlaid in the bottom-right of the output video in real time.

### Behavior

- Every `--court-update-interval` seconds (default 8), the current frame is checked.
- Quality score: >= 2 players visible AND Laplacian variance > 30; the score must be >= `--court-update-min-quality` (default 0.5).
- The update is accepted only if a new detection exists and all corners are within 100 px of the prior model.
- On accept: write `outputs/<run>/court_annotations.txt` and swap the system's `court_mapper`, so subsequent frames re-use the new model.

The heatmap pushes one event per frame (player position in court coordinates). Events older than `--heatmap-window` seconds are evicted. The minimap (240x130) overlays in the bottom-right.

### CLI flags

```bash
python -m badminton_analysis.live --source screen_capture \
       --court-update-interval 5 --heatmap-window 60 --no-heatmap
```

| Flag | Default | Description |
|---|---|---|
| `--court-update-interval` | 8.0 | Seconds between court re-checks |
| `--court-update-min-quality` | 0.5 | Minimum quality score (0-1) |
| `--no-heatmap` | False | Disable the heatmap overlay |
| `--heatmap-window` | 120.0 | Sliding window in seconds |

The Streamlit sidebar "Advanced" expander exposes the same controls.

> Note: `main.py` (local-file analysis) defaults to `enable_court_updater=False` to keep the byte-level regression identical. The updater is only enabled in `live.py` / Streamlit real-time sources.


## Post-analysis

After stopping recognition or completing video processing, the system automatically performs post-analysis on `detections.jsonl`:

- **Rally segmentation**: Splits detection data into individual rallies based on rally IDs
- **KDE heatmaps**: Generates kernel density estimation heatmaps for each rally and the full match
- **Scatter plots**: Player position scatter plots, color-coded by upper/lower court
- **Movement statistics**: Per-rally movement distance, average speed, and maximum speed

Results are saved in `outputs/runs/run_<timestamp>_<id>/post_analysis/`.


## Differences from the Original

This project is a derivative work based on the original badminton analysis tool. Key improvements:

| Aspect | Original | This Project |
|--------|----------|--------------|
| Input sources | Local video files only | Local video + screen capture + headless browser |
| Web UI | Gradio (offline upload) | Streamlit (real-time frame display + interactive annotation) |
| Court model | One-time detection at startup | Auto periodic update + drift correction |
| Heatmap | Post-processing only | Real-time overlay + post-analysis KDE high-quality maps |
| Post-analysis | None | Rally segmentation, KDE heatmaps, scatter plots, movement statistics |
| GPU optimization | CPU default | FP16 + TF32 + cuDNN benchmark |
| Data storage | JSONL + video | JSONL + video + SQLite history + metrics.json |
| Court model file | reference.py (separate module) | Constants inlined into detector.py, reduced module dependencies |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yo-WASSUP/BirdieEye&type=Date)](https://www.star-history.com/#yo-WASSUP/BirdieEye&Date)

