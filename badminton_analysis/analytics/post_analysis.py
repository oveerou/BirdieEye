"""Post-processing analysis: per-rally heatmaps, scatter plots, and movement stats.

Reads detections.jsonl produced during real-time processing and generates
publication-quality visualizations (KDE heatmaps + scatter plots) using
matplotlib/seaborn.  Mirrors the original project's player_positions_zh.py
but packaged as a single callable function.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ── Constants ──
COURT_W = 6.1   # metres
COURT_H = 13.4  # metres
FPS = 30
MAX_SPEED = 8.0  # m/s
MIN_MOVEMENT = 0.05  # m, noise floor
RALLY_GAP_THRESHOLD = 100  # frames
MIN_RALLY_FRAMES = 60
SAMPLE_INTERVAL = 5

UPPER_COLOR = "#ff6363"
LOWER_COLOR = "#63c6ff"
COURT_LINE_COLOR = "#bbbbbb"

plt.style.use("dark_background")

_FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "simhei.ttf")
_FONT_PROP = None
if os.path.isfile(_FONT_PATH):
    import matplotlib.font_manager as fm
    fm.fontManager.addfont(_FONT_PATH)
    _FONT_PROP = fm.FontProperties(fname=_FONT_PATH)
    plt.rcParams["font.family"] = [_FONT_PROP.get_name(), "sans-serif"]
    plt.rcParams["font.sans-serif"] = [_FONT_PROP.get_name()]
plt.rcParams["axes.unicode_minus"] = False


# ── Helpers ──

def _load_detections(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            players = rec.get("players", {})
            upper = (players.get("upper") or {}).get("court") or [None, None]
            lower = (players.get("lower") or {}).get("court") or [None, None]
            rows.append({
                "Frame": rec.get("frame"),
                "Upper_X": upper[0], "Upper_Y": upper[1],
                "Lower_X": lower[0], "Lower_Y": lower[1],
            })
    return pd.DataFrame(rows)


def _segment_rallies(frames: list[int]) -> list[tuple[int, int]]:
    gaps = [frames[i + 1] - frames[i] for i in range(len(frames) - 1)]
    breaks = [i + 1 for i, g in enumerate(gaps) if g > RALLY_GAP_THRESHOLD]
    segments, start = [], 0
    for b in breaks:
        segments.append((start, b))
        start = b
    if start < len(frames):
        segments.append((start, len(frames)))
    return [(s, e) for s, e in segments if e - s >= MIN_RALLY_FRAMES]


def _calc_stats(positions: np.ndarray, times: np.ndarray) -> dict:
    stats = {"total_distance": 0.0, "max_speed": 0.0, "avg_speed": 0.0, "total_frames": len(positions)}
    if len(positions) < 2:
        return stats
    total_dist, max_spd = 0.0, 0.0
    n = len(positions)
    sample_pts = list(range(0, n, SAMPLE_INTERVAL))
    if sample_pts[-1] != n - 1:
        sample_pts.append(n - 1)
    for i in range(1, len(sample_pts)):
        idx_a, idx_b = sample_pts[i - 1], sample_pts[i]
        dx = positions[idx_b][0] - positions[idx_a][0]
        dy = positions[idx_b][1] - positions[idx_a][1]
        d = np.sqrt(dx * dx + dy * dy)
        if d < MIN_MOVEMENT:
            continue
        dt = max(times[idx_b] - times[idx_a], 1e-6)
        spd = d / dt
        if spd > MAX_SPEED:
            continue
        total_dist += d
        max_spd = max(max_spd, spd)
    total_time = max(times[-1] - times[0], 1e-6)
    stats["total_distance"] = round(total_dist, 2)
    stats["max_speed"] = round(max_spd, 2)
    stats["avg_speed"] = round(total_dist / total_time, 2)
    return stats


def _draw_court(ax=None):
    if ax is not None:
        plt.sca(ax)
    plt.gca().invert_yaxis()
    rect = plt.Rectangle((0, 0), COURT_W, COURT_H, fill=False, color=COURT_LINE_COLOR, linewidth=4)
    plt.gca().add_patch(rect)
    sw = 0.46
    sl = 1.98
    bs = 0.76
    plt.plot([sw, sw], [0, COURT_H], COURT_LINE_COLOR, linewidth=4)
    plt.plot([COURT_W - sw, COURT_W - sw], [0, COURT_H], COURT_LINE_COLOR, linewidth=4)
    plt.axhline(y=COURT_H / 2, color=COURT_LINE_COLOR, linestyle="--", linewidth=4)
    plt.plot([COURT_W / 2, COURT_W / 2], [0, COURT_H / 2 - sl], COURT_LINE_COLOR, linewidth=4)
    plt.plot([COURT_W / 2, COURT_W / 2], [COURT_H / 2 + sl, COURT_H], COURT_LINE_COLOR, linewidth=4)
    plt.axhline(y=COURT_H / 2 - sl, color=COURT_LINE_COLOR, linewidth=4)
    plt.axhline(y=COURT_H / 2 + sl, color=COURT_LINE_COLOR, linewidth=4)
    plt.axhline(y=bs, color=COURT_LINE_COLOR, linewidth=4)
    plt.axhline(y=COURT_H - bs, color=COURT_LINE_COLOR, linewidth=4)
    plt.xlim(-0.5, COURT_W + 0.5)
    plt.ylim(COURT_H + 0.5, -0.5)


def _add_stats_text(stats_map: dict, rally_id=None):
    """Add a formatted statistics panel on the right side of the plot."""
    fp = {"fontproperties": _FONT_PROP} if _FONT_PROP else {}

    if rally_id is not None and rally_id in stats_map:
        s = stats_map[rally_id]
        lines = [f"回合 {rally_id} 统计:\n"]
        for half, label in [("upper", "上场球员"), ("lower", "下场球员")]:
            if half in s:
                hs = s[half]
                lines.append(f"  {label}:")
                lines.append(f"    平均速度: {hs['avg_speed']:.2f} 米/秒")
                lines.append(f"    最大速度: {hs['max_speed']:.2f} 米/秒")
                lines.append(f"    移动距离: {hs['total_distance']:.2f} 米\n")
    else:
        lines = ["比赛统计\n"]
        for half, label in [("upper", "上场球员"), ("lower", "下场球员")]:
            dists, speeds, avgs = [], [], []
            for v in stats_map.values():
                if half in v:
                    dists.append(v[half]["total_distance"])
                    if v[half].get("max_speed", 0) > 0:
                        speeds.append(v[half]["max_speed"])
                    if v[half].get("avg_speed", 0) > 0:
                        avgs.append(v[half]["avg_speed"])
            if dists:
                lines.append(f"  {label}:")
                lines.append(f"    总距离: {sum(dists):.2f} 米")
                lines.append(f"    平均速度: {np.mean(avgs):.2f} 米/秒" if avgs else "")
                lines.append(f"    最大速度: {max(speeds):.2f} 米/秒\n" if speeds else "")

    text = "\n".join(line for line in lines if line is not None)
    plt.text(0.98, 0.5, text, ha="right", va="center", transform=plt.gca().transAxes,
             bbox=dict(facecolor="#2a2a2a", alpha=0.85, boxstyle="round,pad=0.8",
                       edgecolor="#555555", linewidth=1.5),
             fontsize=11, color="#ffffff", linespacing=1.6, **fp)


def _make_heatmap(upper_df, lower_df, stats_map, rally_id, save_path):
    plt.figure(figsize=(10, 16), facecolor="#1a1a1a")
    _draw_court()
    upper_cmap = LinearSegmentedColormap.from_list("u", [(0, 0, 0, 0), UPPER_COLOR])
    lower_cmap = LinearSegmentedColormap.from_list("l", [(0, 0, 0, 0), LOWER_COLOR])
    if not upper_df.empty:
        sns.kdeplot(x=upper_df["court_x"], y=upper_df["court_y"],
                     cmap=upper_cmap, fill=True, alpha=1, levels=12, thresh=0.01, bw_adjust=1)
    if not lower_df.empty:
        sns.kdeplot(x=lower_df["court_x"], y=lower_df["court_y"],
                     cmap=lower_cmap, fill=True, alpha=1, levels=12, thresh=0.01, bw_adjust=1)
    _add_stats_text(stats_map, rally_id)
    fp = {"fontproperties": _FONT_PROP} if _FONT_PROP else {}
    title = "球员位置热力图"
    subtitle = f"回合 {rally_id}" if rally_id else "全场汇总"
    plt.title(f"{title}\n{subtitle}", color="white", fontsize=15, **fp)
    plt.xlabel("场地宽度（米）", color="white", fontsize=12, **fp)
    plt.ylabel("场地长度（米）", color="white", fontsize=12, **fp)
    plt.tick_params(colors="white", labelsize=10)
    plt.xlim(0, COURT_W)
    plt.ylim(COURT_H, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()


def _make_scatter(upper_df, lower_df, save_path, rally_id=None):
    plt.figure(figsize=(10, 16), facecolor="#1a1a1a")
    _draw_court()
    if not upper_df.empty:
        plt.scatter(upper_df["court_x"], upper_df["court_y"], alpha=0.6, s=20, color=UPPER_COLOR)
    if not lower_df.empty:
        plt.scatter(lower_df["court_x"], lower_df["court_y"], alpha=0.6, s=20, color=LOWER_COLOR)
    fp = {"fontproperties": _FONT_PROP} if _FONT_PROP else {}
    title = "球员位置散点图"
    subtitle = f"回合 {rally_id}" if rally_id else "全场汇总"
    plt.title(f"{title}\n{subtitle}", color="white", fontsize=15, **fp)
    plt.xlabel("场地宽度（米）", color="white", fontsize=12, **fp)
    plt.ylabel("场地长度（米）", color="white", fontsize=12, **fp)
    plt.tick_params(colors="white", labelsize=10)
    plt.xlim(0, COURT_W)
    plt.ylim(COURT_H, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()


# ── Public API ──

def run_post_analysis(detections_path: str, output_dir: str, fps: int = 30) -> dict:
    """Run full post-processing analysis.

    Args:
        detections_path: path to detections.jsonl
        output_dir: directory for generated images
        fps: video frame rate

    Returns:
        dict with keys:
          - heatmap_paths: list[str]
          - scatter_paths: list[str]
          - rally_stats: dict  (rally_id -> {upper/lower -> stats})
          - match_stats: dict  (same structure, aggregated)
          - num_rallies: int
    """
    if not os.path.isfile(detections_path):
        print(f"[post_analysis] detections file not found: {detections_path}")
        return {"heatmap_paths": [], "scatter_paths": [], "rally_stats": {}, "match_stats": {}, "num_rallies": 0}

    df = _load_detections(detections_path)
    if df.empty or "Frame" not in df.columns:
        return {"heatmap_paths": [], "scatter_paths": [], "rally_stats": {}, "match_stats": {}, "num_rallies": 0}

    frames = df["Frame"].astype(int).tolist()
    rally_segments = _segment_rallies(frames)

    heatmap_dir = os.path.join(output_dir, "heatmaps")
    scatter_dir = os.path.join(output_dir, "scatter_plots")
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(scatter_dir, exist_ok=True)

    def _build_half_df(raw_x_col, raw_y_col):
        sub = df[[raw_x_col, raw_y_col]].copy()
        sub.columns = ["court_x", "court_y"]
        sub["valid"] = sub["court_x"].notna() & sub["court_y"].notna() & (sub["court_x"] >= 0) & (sub["court_y"] >= 0)
        return sub

    # ── Compute per-rally stats ──
    rally_stats: dict = {}
    frame_times = np.array(frames) / fps

    for rid, (s, e) in enumerate(rally_segments, 1):
        r_times = frame_times[s:e]
        for half, xc, yc in [("upper", "Upper_X", "Upper_Y"), ("lower", "Lower_X", "Lower_Y")]:
            sub = _build_half_df(xc, yc).iloc[s:e]
            valid = sub[sub["valid"]]
            if len(valid) > 1:
                rally_stats.setdefault(rid, {})[half] = _calc_stats(
                    valid[["court_x", "court_y"]].values, r_times[valid.index - s]
                )

    # ── Per-rally visualizations ──
    heatmap_paths = []
    scatter_paths = []

    for rid, (s, e) in enumerate(rally_segments, 1):
        upper_sub = _build_half_df("Upper_X", "Upper_Y").iloc[s:e]
        lower_sub = _build_half_df("Lower_X", "Lower_Y").iloc[s:e]
        upper_valid = upper_sub[upper_sub["valid"]]
        lower_valid = lower_sub[lower_sub["valid"]]

        hp = os.path.join(heatmap_dir, f"rally_{rid}_heatmap.png")
        _make_heatmap(upper_valid, lower_valid, rally_stats, rid, hp)
        heatmap_paths.append(hp)

        sp = os.path.join(scatter_dir, f"rally_{rid}_scatter.png")
        _make_scatter(upper_valid, lower_valid, sp, rally_id=rid)
        scatter_paths.append(sp)

    # ── Match-wide visualizations ──
    all_upper = _build_half_df("Upper_X", "Upper_Y")
    all_lower = _build_half_df("Lower_X", "Lower_Y")
    # Only use frames that belong to valid rallies
    rally_frame_mask = np.zeros(len(df), dtype=bool)
    for s, e in rally_segments:
        rally_frame_mask[s:e] = True
    match_upper = all_upper[all_upper["valid"] & rally_frame_mask]
    match_lower = all_lower[all_lower["valid"] & rally_frame_mask]

    match_hp = os.path.join(heatmap_dir, "match_heatmap.png")
    _make_heatmap(match_upper, match_lower, rally_stats, None, match_hp)
    heatmap_paths.append(match_hp)

    match_sp = os.path.join(scatter_dir, "match_scatter.png")
    _make_scatter(match_upper, match_lower, match_sp)
    scatter_paths.append(match_sp)

    # ── Aggregate match-wide stats ──
    match_stats: dict = {"upper": {"total_distance": 0.0, "max_speed": 0.0, "avg_speed": 0.0},
                         "lower": {"total_distance": 0.0, "max_speed": 0.0, "avg_speed": 0.0}}
    for half in ("upper", "lower"):
        dists, speeds, avgs = [], [], []
        for rs in rally_stats.values():
            if half in rs:
                dists.append(rs[half]["total_distance"])
                if rs[half].get("max_speed", 0) > 0:
                    speeds.append(rs[half]["max_speed"])
                if rs[half].get("avg_speed", 0) > 0:
                    avgs.append(rs[half]["avg_speed"])
        if dists:
            match_stats[half]["total_distance"] = round(sum(dists), 2)
            match_stats[half]["max_speed"] = round(max(speeds), 2) if speeds else 0.0
            match_stats[half]["avg_speed"] = round(float(np.mean(avgs)), 2) if avgs else 0.0

    print(f"[post_analysis] Generated {len(heatmap_paths)} heatmaps, {len(scatter_paths)} scatter plots")
    return {
        "heatmap_paths": heatmap_paths,
        "scatter_paths": scatter_paths,
        "rally_stats": rally_stats,
        "match_stats": match_stats,
        "num_rallies": len(rally_segments),
    }
