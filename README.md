# BirdieEye: AI 羽毛球鹰眼系统 🏸

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/yo-WASSUP/BirdieEye?style=social)](https://github.com/yo-WASSUP/BirdieEye/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yo-WASSUP/BirdieEye?style=social)](https://github.com/yo-WASSUP/BirdieEye/network/members)
[![GitHub license](https://img.shields.io/github/license/yo-WASSUP/BirdieEye)](https://github.com/yo-WASSUP/BirdieEye/blob/main/LICENSE)
[![小红书视频介绍](https://img.shields.io/badge/小红书-视频介绍-ff2442)](https://www.xiaohongshu.com/explore/6a37b1d20000000011016229?xsec_token=ABod3wXBTiDppp6W2Ou0QHlu2eotUkeu27-ha64nFRR74=&xsec_source=pc_user)

**基于计算机视觉的羽毛球比赛视频分析工具**

[中文](README.md) | [English](README_EN.md)

</div>

## 🎬 效果预览

| 功能 | 截图 |
|------|------|
| **屏幕实时识别** — 捕获任意屏幕内容进行实时分析 | ![屏幕识别](screenshots/demo-1.png) |
| **球场角点标注** — 通过 4 个角点定位球场范围，建立坐标映射 | ![角点标注](screenshots/demo-2.png) |
| **实时运行画面** — 骨架绘制、球员追踪、轨迹叠加 | ![运行中](screenshots/demo-3.png) |
| **赛后分析** — 全场/每回合热力图、散点图、移动统计 | ![赛后分析](screenshots/demo-4.png) |

## 🆕 更新日志
- **2026-06-29**：新增后期分析模块（回合切片、KDE 热力图、散点图、移动统计）；球场漂移矫正；GPU FP16/TF32 推理优化；Web UI 交互改进（错误堆栈展示、球场检测重试、启动提醒）。
- **2026-06-27**：新增多源实时输入（屏幕捕获 / 无头浏览器）；Streamlit Web 控制台；球场模型自动更新 + 实时热力图；SQLite 历史记录。
- **2026-06-23**：增加自动球场边界检测。
- **2026-06-20**：正式开源。
- **2026-06-17**：整理项目介绍文档。
- **当前版本**：支持球员姿态检测、羽毛球检测、球场坐标映射、轨迹统计、热力图/散点图和带标注视频输出。
- **实验功能**：击球点分析和技术动作统计仍在迭代中，适合研究和二次开发使用。

## 🔮 开发计划

- [x] 羽毛球比赛视频逐帧分析
- [x] RTMPose / RTMO / YOLO Pose 多姿态模型支持
- [x] YOLO 羽毛球检测模型接入
- [x] 手动球场标注与球场坐标映射
- [x] 球员移动轨迹、速度、距离和回合统计
- [x] 中文 / 英文可视化文字
- [x] 热力图、散点图和检测数据导出
- [x] 自动球场关键点检测
- [x] 多源实时输入（屏幕捕获 / 无头浏览器 / 本地视频）
- [x] Streamlit Web 控制台（一键启动 start.bat）
- [x] 球场模型自动更新 + 漂移矫正
- [x] 实时热力图叠加
- [x] 后期分析（回合切片、KDE 热力图、散点图）
- [x] GPU FP16 / TF32 推理优化
- [x] SQLite 历史记录与指标存储
- [ ] 更稳定的击球点识别
- [ ] 更精确的羽毛球检测模型
- [ ] 更完整的技术动作统计
- [ ] 批量视频分析工作流

---

## ✨ 功能

- **球员姿态检测** - 支持 RTMPose、RTMO 和 Ultralytics YOLO Pose，识别人体关键点和骨架。
- **羽毛球检测** - 使用 YOLO 模型检测羽毛球位置，并在输出视频中绘制轨迹。
- **球场坐标映射** - 手动标注球场关键点，将图像坐标映射到标准球场坐标。
- **球员位置追踪** - 分别追踪上半场和下半场球员位置，记录移动轨迹。
- **回合检测** - 根据连续球场视图自动判断回合开始和结束，并在视频叠加层和检测数据中记录回合编号。
- **运动统计分析** - 统计移动距离、当前速度、最大速度和回合数量。
- **可视化输出** - 生成带骨架、轨迹、统计信息和球场轨迹的分析视频。
- **位置图表** - 自动生成球员位置热力图和散点图。
- **中英文显示** - 可通过 `--language zh/en` 切换可视化文字。
- **本地运行** - 视频、模型和分析结果都保存在本地。
- **多源实时输入** - 支持本地视频、屏幕捕获（mss）和无头浏览器（Chrome DevTools Protocol）三种输入源，可分析直播画面。
- **球场漂移矫正** - 通过单应性矩阵定期重新检测球场角点，修正长时间运行中的摄像头漂移，保持坐标映射精度。
- **实时热力图** - 2 分钟滑动窗口热力图，上下半场分图，实时叠加在视频右下角小地图中。
- **后期分析** - 自动回合切片、KDE 热力图、散点图和每回合移动统计，生成高质量分析图表。
- **GPU 推理优化** - FP16 半精度推理 + TF32 矩阵乘法 + cuDNN benchmark，显著提升 GPU 帧率。
- **Web 控制台** - Streamlit 界面，支持实时帧显示、参数调节、球场标注、历史记录查看，一键 `start.bat` 启动。
- **SQLite 历史** - 每次运行自动记录指标到 SQLite 数据库，支持历史回溯和对比分析。

## 📋 系统要求

- Python 3.8+
- 字体：中文显示需要黑体字体文件（如 `simhei.ttf`）。请手动从 [GitHub Releases](https://github.com/yo-WASSUP/BirdieEye/releases/latest) 下载并放置到项目根目录。

- FFmpeg，并已加入系统 `PATH`
- 羽毛球 YOLO 检测权重，请从 [GitHub Releases](https://github.com/yo-WASSUP/BirdieEye/releases/latest)  下载

## 性能需求与参考速度

推荐配置：

- GPU，建议 6GB+ 显存；显存越大，越适合更高分辨率视频和更大的姿态模型。
- 16GB+ 系统内存。
- SSD 存储，方便写入输出视频、`detections.jsonl` 和可视化图片。
- CPU 可以运行完整流程，但姿态检测和羽毛球检测会明显变慢，更适合短视频或功能验证。

参考速度会受显卡、视频分辨率、姿态模型、是否显示窗口、是否保留音频影响。

以 720p 视频、`--pose-family yolo-pose --yolo-pose-model yolo11n-pose.pt` 和 `weights/yolo11s-ball.pt` 为例，GPU 推理日志通常接近：

```text
pose 0.02s, shuttlecock 0.02s, shuttle draw 0.00s, players draw 0.01s, court draw 0.00s
```

开启 `--performance-stats` 可以每隔约 5 秒打印一次性能汇总，用于判断瓶颈在姿态推理、羽毛球检测还是绘制阶段。

## 🚀 安装指南

默认依赖使用 CPU 版 PyTorch 和 ONNX Runtime。

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

### GPU 加速（Windows / NVIDIA）

前置要求：

- 已安装 NVIDIA 显卡驱动，`nvidia-smi` 可以正常输出显卡信息。
- 推荐使用 CUDA 12.1 对应的 PyTorch wheel。

PowerShell：

```bash
.\.venv\Scripts\activate

pip uninstall -y torch torchvision onnxruntime onnxruntime-gpu
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu==1.20.1
```

验证 GPU 是否生效：

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available')"
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

期望看到：

```text
cuda: True
CUDAExecutionProvider
```

> 注意：安装 GPU 版 ONNX Runtime 后，`pip check` 可能提示 `rtmlib requires onnxruntime, which is not installed`。只要 provider 验证能看到 `CUDAExecutionProvider`，就不要再安装 CPU 版 `onnxruntime`，否则可能覆盖 GPU 包。

切回 CPU 版：

```bash
pip install --force-reinstall -r requirements.txt
```

## 📝 使用指南

### 第一次运行流程

1. 准备输入视频和羽毛球检测权重。
2. 运行基础命令：

```bash
python main.py --video-path videos/demo.mp4
```

3. 如果没有传 `--template-path`，程序会弹出文件选择框，让你选择一张球场模板图。模板图通常选视频里视角稳定、球场线清楚的一帧。
4. 程序会先尝试自动检测球场边界，并保存 `outputs/<视频文件名>/auto_court_preview.png`。预览窗口按 Enter/Y 接受自动结果；按 M/R/Esc 进入手动四角标注。
5. 如果进入手动标注，按图片顶部提示依次点击球场四个角点：左上、右上、右下、左下。

6. 点完四个点后，窗口会显示绿色球场框和蓝色姿态检测 ROI 框。ROI 由程序根据球场自动生成。
7. 标注结果会保存到 `outputs/<视频文件名>/court_annotations.txt`。同一个输出目录下再次运行会复用这个文件，不会重复要求标注。
8. 分析结束后，查看 `outputs/<视频文件名>/detect_<视频文件名>.mp4`、`detections.jsonl` 和 `position_visualizations/`。

为什么要标注球场四点：

- 四个角点用于建立图像坐标到标准羽毛球场坐标的映射。
- 球员过滤主要依赖球场坐标，能把观众、裁判、场外人员过滤掉。
- 上下半场球员判断、移动距离、速度、回合统计、热力图和散点图都依赖这个映射。
- 回合检测基于球场模板匹配：连续多帧识别为比赛视图时开始回合，连续多帧离开比赛视图时结束回合。
- 姿态检测 ROI 只用于减少推理区域和提升速度；它会自动从球场范围扩展生成。
- 羽毛球检测仍在整帧上执行，轨迹显示会按球场横向范围加 padding 做基础过滤。

如果你换了视频视角、裁切方式或模板图，需要删除对应输出目录里的 `court_annotations.txt`，重新标注四点。

### 姿态模型选择

```bash
# 默认：两阶段 RTMPose balanced
python main.py --video-path videos/demo.mp4 --pose-family rtmpose --pose-mode balanced

# 更轻量的一阶段 RTMO
python main.py --video-path videos/demo.mp4 --pose-family rtmo --pose-mode lightweight

# 使用 Ultralytics YOLO Pose
python main.py --video-path videos/demo.mp4 --pose-family yolo-pose --yolo-pose-model yolo11n-pose.pt
```

RTMPose 模型档位：

- `lightweight`：速度优先。
- `balanced`：默认配置，速度和效果折中。
- `performance`：更大模型，速度更慢，通常更适合追求检测质量。

### 常用参数

```text
--video-path                 输入视频路径，必填
--output-dir                 输出目录，默认 outputs/<视频文件名>
--ball-model                 YOLO 羽毛球检测模型路径，默认 weights/yolo11s-ball.pt
--pose-family                姿态模型族：rtmpose、rtmo 或 yolo-pose
--pose-mode                  RTMPose / RTMO 档位：lightweight、balanced、performance
--yolo-pose-model            YOLO pose 模型路径或模型名，默认 yolo11n-pose.pt
--template-path              球场模板图路径；不传时会弹出文件选择框
--pose-roi true|false                是否显示姿态检测 ROI 框，默认 true
--display true|false                 是否显示 OpenCV 预览窗口，默认 true
--skeletons true|false               是否显示人体骨架，默认 true
--player-trajectories true|false     是否显示球员轨迹，默认 true
--court-trajectory true|false        是否显示球场轨迹叠加层，默认 true
--shuttlecock-trajectory true|false  是否显示羽毛球轨迹，默认 true
--player-stats true|false            是否显示球员统计信息，默认 true
--performance-stats                  打印性能耗时
--save-images                        保存处理后的每帧图像
--visualize-positions true|false     是否生成热力图和散点图，默认 true
--audio true|false                   是否保留原视频音频，默认 true
--language {zh,en}           选择界面语言
```

## 📊 输出结果

默认输出到 `outputs/<视频文件名>/`：

- `metadata.json`：视频、模型、球场标注和输出文件元数据。
- `detections.jsonl`：逐帧检测记录，包含回合编号、球员、手部、球场坐标、速度和羽毛球坐标。
- `detect_<视频文件名>.mp4`：带骨架、轨迹、统计信息和回合编号叠加层的输出视频。
- `court_annotations.txt`：球场标注坐标缓存。
- `position_visualizations/heatmaps/`：球员位置热力图。
- `position_visualizations/scatter_plots/`：球员位置散点图。

## 🧩 项目结构

```text
main.py              # 命令行入口（本地视频分析）
app.py               # Streamlit Web 控制台入口
start.bat            # 一键启动脚本（自动建 venv + 装依赖 + 起 Streamlit）
badminton_analysis/
├── system.py        # 视频分析主流程 BadmintonAnalysisSystem
├── config.py        # YAML 配置加载与 AppConfig
├── storage.py       # SQLite 历史 + metrics.json 写入
├── analytics/       # 实时热力图 + 后期分析（回合切片、KDE、散点图）
├── sources/         # 多源输入（屏幕捕获 / 无头浏览器 / 视频文件 / 流适配器）
├── court/           # 球场检测、坐标映射、漂移矫正、自动更新
├── detection/       # 羽毛球检测与姿态检测
├── tracking/        # 球员追踪
├── visualization/   # 视频叠加层、统计图和位置图
└── media/           # 视频音频处理
```

## 🙏 致谢

感谢 RTMPose、RTMO 和 OpenMMLab 生态提供的姿态估计算法基础，以及 [Tau-J/rtmlib](https://github.com/Tau-J/rtmlib) 提供的轻量姿态估计运行库。

感谢 [Ultralytics](https://github.com/ultralytics/ultralytics) 提供的 YOLO 目标检测算法与工具链。

感谢 [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) 项目整理并公开羽毛球数据集，为本项目的羽毛球检测与轨迹分析提供了重要参考。

## 📄 许可证

本项目代码和 `weights/yolo11s-ball.pt` 使用 Apache License 2.0。随 Release 提供的 RTMPose / RTMO / YOLOX ONNX 权重来自 OpenMMLab / RTMPose 生态，按其上游 Apache License 2.0 授权使用，并保留原始归属。

## 实时源 (Real-time sources)

除了本地视频文件，也可以直接从屏幕区域或网页直播获取画面。两者都通过新入口 `python -m badminton_analysis.live` 启动，**不会改动**现有的 `python main.py --video-path xxx.mp4` 流程。

新增的依赖：`mss`（屏幕捕获）、`websocket-client` + `requests`（无头浏览器）。已在 `requirements.txt` 中固定。

### 屏幕捕获 (mss)

抓取屏幕指定区域，可用于分析本地视频播放器窗口、训练软件等。

```bat
python -m badminton_analysis.live --source screen_capture --region 100,100,1280,720 --fps 30 --display true
```

参数 `--region left top width height` 指定抓取矩形，默认 `100 100 1280 720`。运行时会自动抓首帧作为球场标注源（自动检测或手动点 4 角点）。

### 网页直播 (无头浏览器)

用 headless Chrome 打开含 `<video>` 元素的网页，通过 Chrome DevTools Protocol 抓取 video 帧。**不依赖 selenium**，只用 `subprocess + websocket-client`。

```bat
python -m badminton_analysis.live --source browser_headless --url "https://example.com/live" --fps 30 --display true
```

前置：系统已安装 Google Chrome（默认路径 `C:\Program Files\Google\Chrome\Application\chrome.exe`，可通过 `--chrome-path` 覆盖）。其他参数：`--wait-sec 20` 增大等待时间、`--browser-w/--browser-h` 设置窗口大小。

### 停止与输出

按 `Ctrl+C` 即可优雅停止。结束后输出在 `outputs/live_<source>_<timestamp>/detect_*.mp4`（无音频；同 `keep_audio=false` 路径）。

### 已知限制

- 输出 mp4 的 fps 固定为 `--fps` 参数（默认 30），与实际源帧率不一致时播放速度会有偏差
- 首帧是球场标注的来源，会出现在输出视频开头
- Chrome 路径非默认时需用 `--chrome-path` 显式指定
- macOS / Linux 屏幕捕获未经测试（mss 库支持但需自行验证）


## Web 控制台 (Streamlit)

除了 CLI（`main.py` / `live.py`），也提供 Streamlit Web 控制台。结构参考 `football-realtime-analyzer`（侧栏参数 + 主区实时画面 + 底部历史），检测层用羽毛球原生（YOLO Pose + 球检测 + 球场映射 + 上下半场球员追踪）。

### 启动

```bat
:: 一键启动（首次会自动建 venv + 装依赖 + 起 Streamlit）
start.bat

:: 或手动
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501
```

浏览器打开 http://localhost:8501

### 视频源

侧栏选择三类源之一：
- **本地视频**：上传 mp4/avi/mov（最大 4GB）
- **网页直播**：粘 https URL，无头 Chrome 抓 `<video>` 元素
- **屏幕捕获**：预设（全屏 / 左半 / 右半 / 上半 / 下半 / 中央 1280x720）或自定义矩形，可"预览捕获区域"

### 实时统计

右栏每帧刷新：
- FPS / 帧号 / 设备
- 检测球员数 / 羽毛球可见
- 当前回合号
- 上半场 + 下半场球员数 / 平均速度 / 累计移动距离

跑完写入 SQLite `outputs/badminton.db`，底部表格可看历史。

### 球场检测选项

- 默认：自动检测（需画面有清晰球场线）+ 非交互模式
- "跳过球场自动检测" 复选框：--no-court 模式，对任意内容都能跑（无球场映射）

### 文件位置

| 目录 | 内容 |
|---|---|
| `outputs/runs/run_<时间戳>_<id>/` | 每轮次的 `metrics.json` + 标注视频 |
| `outputs/badminton.db` | SQLite 历史 |
| `outputs/uploads/` | Web 上传的本地视频 |

### 已知限制

- 屏幕源非交互式（无 GUI 弹窗），所以"跳过球场自动检测"是屏幕源推荐选项
- Streamlit 单用户；多人用需要鉴权层
- 跑长视频建议提高 `frame_skip` 节省 GPU


## 球场模型自动更新 + 实时热力图

比赛过程中，球员的"全场画面"（两人清晰可见 + 帧清晰）会用来自动刷新球场模型；2 分钟滑动窗口热力图（上下半场分图）实时叠加在 output 视频右下角。

### 行为

- 每 `--court-update-interval` 秒（默认 8 秒）检查一次当前帧
- 质量评分：≥2 个球员 + Laplacian 方差 > 30；分数需 ≥ `--court-update-min-quality`（默认 0.5）
- 通过质量 + 角点距离合理性（任意角点偏离 < 100px）才接受
- 接受后写 `outputs/<run>/court_annotations.txt` 持久化，并立即更新 `court_mapper`，让后续帧的姿态映射跟随新模型

热力图每个 frame 把球员的 court 坐标位置入队，过期（> `--heatmap-window` 秒）自动剔除。上半场、下半场各画一张，合成一张 240x130 minimap 叠加到画面右下角。

### CLI 标志

```bash
python -m badminton_analysis.live --source screen_capture \
       --court-update-interval 5 --heatmap-window 60 --no-heatmap
```

| 标志 | 默认 | 说明 |
|---|---|---|
| `--court-update-interval` | 8.0 | 球场模型重新检查间隔（秒） |
| `--court-update-min-quality` | 0.5 | 最低质量分数（0-1） |
| `--no-heatmap` | False | 关闭热力图叠加 |
| `--heatmap-window` | 120.0 | 热力图滑动窗口（秒） |

Streamlit 侧栏"高级"折叠区可调。

> 注：`main.py`（本地文件分析）默认 `enable_court_updater=False`，保持与原版 byte-level 一致；球场自动更新只在 `live.py` / Streamlit 实时源里启用。


## 后期分析

识别停止或视频处理完成后，自动对 `detections.jsonl` 进行后期分析：

- **回合切片**：根据回合编号将检测数据切分为独立回合
- **KDE 热力图**：为每个回合和全场生成核密度估计热力图
- **散点图**：球员位置散点图，按上下半场分色
- **移动统计**：每回合移动距离、平均速度、最大速度

结果保存在 `outputs/runs/run_<时间戳>_<id>/post_analysis/` 目录下。


## 与原版的差异

本项目基于原版羽毛球分析工具二次开发，主要改进如下：

| 方面 | 原版 | 本项目 |
|------|------|--------|
| 输入源 | 仅本地视频文件 | 本地视频 + 屏幕捕获 + 无头浏览器 |
| Web UI | Gradio（离线上传） | Streamlit（实时帧显示 + 交互标注） |
| 球场模型 | 启动时一次性检测 | 自动定期更新 + 漂移矫正 |
| 热力图 | 仅后期生成 | 实时叠加 + 后期 KDE 高质量图 |
| 后期分析 | 无 | 回合切片、KDE 热力图、散点图、移动统计 |
| GPU 优化 | CPU 默认 | FP16 + TF32 + cuDNN benchmark |
| 数据存储 | JSONL + 视频 | JSONL + 视频 + SQLite 历史 + metrics.json |
| 球场模型文件 | reference.py 独立文件 | 常量内联到 detector.py，减少模块依赖 |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yo-WASSUP/BirdieEye&type=Date)](https://www.star-history.com/#yo-WASSUP/BirdieEye&Date)