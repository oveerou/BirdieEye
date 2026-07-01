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
| **屏幕实时识别** — 捕获任意屏幕内容进行实时分析 | ![屏幕识别](screenshots/demo-3.png) |
| **球场角点标注** — 通过 4 个角点定位球场范围，建立坐标映射 | ![角点标注](screenshots/demo-1.png) |
| **实时运行画面** — 骨架绘制、球员追踪、轨迹叠加 | ![运行中](screenshots/demo-2.png) |
| **赛后分析** — 全场/每回合热力图、散点图、移动统计 | ![赛后分析](screenshots/demo-4.png) |

## 🆕 更新日志

基于原版羽毛球分析工具二次开发，新增多源实时输入（屏幕捕获/无头浏览器）、Streamlit Web 控制台、球场模型自动更新与漂移矫正、实时热力图叠加、后期分析（回合切片/KDE 热力图/散点图）及 GPU FP16/TF32 推理优化。
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

安装依赖：
```bash
pip install -r requirements.txt     # CPU 版（默认）
# GPU 加速需手动换 CUDA 版 torch 和 onnxruntime-gpu
```

模型权重从 [Releases](https://github.com/oveerou/BirdieEye/releases) 下载，放 `weights/` 目录。

## 🚀 快速使用

**本地视频分析（CLI）：**
```bash
python main.py --video-path 视频.mp4
# 首次运行会弹出窗口标注球场 4 个角点，之后会自动复用
```

**实时屏幕捕获：**
```bash
python -m badminton_analysis.live --source screen_capture --region 0,0,1920,1080
```

**Web 控制台（推荐）：**
```bash
start.bat        # 一键启动
# 或手动: python -m streamlit run app.py
# 浏览器打开 http://localhost:8501
```

Web 控制台支持三种输入源：上传视频、屏幕捕获、无头浏览器抓取网页直播。

## 📊 功能概览

| 功能 | 说明 |
|------|------|
| 姿态检测 | YOLO-Pose / RTMPose / RTMO 三引擎，COCO 17 关键点 |
| 羽毛球检测 | YOLO 模型 + 轨迹跟踪 + 异常跳跃过滤 |
| 球场标定 | 自动 HSV 检测 / Hough 线检测 / 手动 4 角点标注 |
| 球员追踪 | 按中线上下半场分流，双坐标系（图像+球场） |
| 回合检测 | 基于模板匹配的滚动窗口（连续 5 帧判定） |
| 运动统计 | 速度、距离、极值，支持回合/全场维度 |
| 实时叠加 | 骨架、轨迹、统计面板、小地图轨迹、2min 热力图 |
| 后期分析 | 回合自动切片、KDE 热力图、散点图、移动统计 |
| 漂移矫正 | 单应性矩阵定期重新检测角点，补偿摄像机移动 |
| GPU 优化 | FP16 推理、TF32 矩阵乘法、cuDNN benchmark |
| 数据存储 | JSONL 逐帧检测记录 + SQLite 历史 + metrics.json |

## 📁 输出文件

```
outputs/<视频名>/
├── detect_*.mp4           # 标注视频（骨架/轨迹/统计）
├── detections.jsonl       # 逐帧检测数据
├── metadata.json          # 运行元信息
├── court_annotations.txt  # 球场角点缓存
└── heatmaps + scatter_plots/   # 热力图与散点图
```

## 📄 许可证

Apache License 2.0
