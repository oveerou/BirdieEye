import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont


class StatsVisualizer:
    """
    统计信息可视化器 - 卡片式面板设计
    """

    def __init__(self, frame_width, frame_height, language='zh'):
        self.language = language
        self.frame_width = frame_width
        self.frame_height = frame_height

        # 缩放因子 (参考1920x1080)
        sf = max(0.7, min(frame_width / 1920.0, frame_height / 1080.0))
        self.sf = sf

        # 卡片尺寸
        self.card_width = int(240 * sf)
        self.card_padding = int(12 * sf)
        self.header_height = int(34 * sf)
        self.corner_radius = int(10 * sf)

        # 字体大小 (像素)
        self.font_header = int(17 * sf)
        self.font_speed = int(26 * sf)
        self.font_label = int(13 * sf)
        self.font_value = int(16 * sf)
        self.font_badge = int(14 * sf)

        # 颜色 (RGBA)
        self.shadow_rgb = (0, 0, 0)
        self.card_bg_rgba = (15, 15, 25, 190)
        self.text_white = (255, 255, 255)
        self.text_dim = (170, 170, 180)
        self.separator = (255, 255, 255, 50)
        self.upper_rgb = (255, 230, 0)       # 黄色
        self.lower_rgb = (255, 0, 200)       # 品红色
        self.badge_bg_rgb = (255, 140, 0)    # 橙色

        # 字体
        self.font_path = self._find_font()
        self._cache = {}

        # 语言文本
        self.t = {
            'zh': {
                'rally': '回合',
                'upper': '上场球员',
                'lower': '下场球员',
                'speed': '当前速度',
                'rally_label': '回合',
                'distance': '距离',
                'avg': '均速',
                'max': '极速',
                'match': '全场',
                'unit_s': 'm/s',
                'unit_d': 'm',
            },
            'en': {
                'rally': 'Rally',
                'upper': 'Upper',
                'lower': 'Lower',
                'speed': 'Speed',
                'rally_label': 'Rally',
                'distance': 'Dist',
                'avg': 'Avg',
                'max': 'Max',
                'match': 'Match',
                'unit_s': 'm/s',
                'unit_d': 'm',
            }
        }[language]

    # ── 字体 ──────────────────────────────────────────────

    def _find_font(self):
        for p in [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/simkai.ttf",
        ]:
            if os.path.exists(p):
                return p
        return None

    def _font(self, size):
        if size not in self._cache:
            self._cache[size] = ImageFont.truetype(self.font_path, size) \
                if self.font_path else ImageFont.load_default()
        return self._cache[size]

    # ── 公开接口 ──────────────────────────────────────────

    def draw_player_stats(self, frame, movement_stats, rally_count):
        """在帧上绘制球员统计面板（主入口）"""
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)

        # 回合徽章 (左上角)
        self._draw_badge(draw, rally_count)

        # 上场球员卡片
        y_upper = max(int(10 * self.sf), int(40 * self.sf))
        upper_h = self._draw_card(pil, draw,
                        self.margin, y_upper,
                        self.t['upper'], movement_stats.get('upper', {}),
                        self.upper_rgb)

        # 下场球员卡片 (紧贴上场卡片底部)
        y_lower = y_upper + upper_h
        lower_h = self._draw_card(pil, draw,
                        self.margin, y_lower,
                        self.t['lower'], movement_stats.get('lower', {}),
                        self.lower_rgb)

        frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def add_text(self, frame, text, position, font_scale, color, thickness):
        """兼容旧调用的文本绘制方法"""
        if self.language == 'en' or self.font_path is None:
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil_img)
            d.text(position, text, font=self._font(int(font_scale * 30)),
                   fill=(color[2], color[1], color[0]))
            frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ── 内部绘制 ──────────────────────────────────────────

    @property
    def margin(self):
        return int(12 * self.sf)

    def _draw_badge(self, draw, rally_count):
        """回合数徽章"""
        text = f"{self.t['rally']} #{rally_count}"
        font = self._font(self.font_badge)
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        px, py = int(10 * self.sf), int(5 * self.sf)
        x, y = self.margin, max(2, int(4 * self.sf))
        draw.rounded_rectangle(
            [x, y, x + tw + px * 2, y + th + py * 2],
            radius=int(7 * self.sf),
            fill=self.badge_bg_rgb,
        )
        draw.text((x + px, y + py), text, font=font,
                  fill=(0, 0, 0), anchor='lt')

    def _draw_card(self, img, draw, x, y, name, stats, accent):
        """绘制单个球员卡片"""
        w = self.card_width
        pad = self.card_padding
        hdr = self.header_height
        r = self.corner_radius

        # ── 计算卡片高度 ──
        speed_area = self.font_speed + int(14 * self.sf)
        section_gap = int(8 * self.sf)
        line_h = self.font_value + int(8 * self.sf)
        label_h = self.font_label + int(6 * self.sf)
        sep_space = int(6 * self.sf)

        h = (pad
             + hdr + int(6 * self.sf)          # header + gap
             + speed_area                       # big speed
             + sep_space + 2                    # separator
             + section_gap                      # section gap
             + label_h                          # "回合" label
             + line_h                           # 距离 + 均速
             + line_h                           # 极速
             + section_gap                      # gap
             + sep_space + 2                    # separator
             + section_gap                      # section gap
             + label_h                          # "全场" label
             + line_h                           # 总距离
             + pad)

        # ── 阴影 (直接画在 img 上, 不透明) ──
        so = int(5 * self.sf)
        draw.rounded_rectangle(
            [x + so, y + so, x + w + so, y + h + so],
            radius=r, fill=(0, 0, 0))

        # ── 半透明卡片背景 (alpha compositing) ──
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rounded_rectangle(
            [x, y, x + w, y + h],
            radius=r, fill=self.card_bg_rgba)
        img.paste(Image.composite(overlay, img.convert('RGBA'),
                                  overlay.split()[3]).convert('RGB'))

        # ── 不透明元素直接画在 img 上 ──
        d2 = ImageDraw.Draw(img)

        # 头部色块 (圆角顶部)
        d2.rounded_rectangle([x, y, x + w, y + hdr], radius=r, fill=accent)
        # 填平底部圆角
        d2.rectangle([x, y + hdr - r, x + w, y + hdr], fill=accent)

        # 头部文字
        d2.text((x + pad, y + (hdr - self.font_header) // 2),
                name, font=self._font(self.font_header),
                fill=(0, 0, 0), anchor='lt')

        # ── 当前速度 (大号) ──
        sy = y + pad + hdr + int(4 * self.sf)
        spd = stats.get('current_speed', 0)
        d2.text((x + pad, sy), f"{spd:.2f}",
                font=self._font(self.font_speed),
                fill=self.text_white, anchor='lt')
        # 单位 (速度右侧)
        spd_font = self._font(self.font_speed)
        spd_bbox = spd_font.getbbox(f"{spd:.2f}")
        spd_tw = spd_bbox[2] - spd_bbox[0]
        d2.text((x + pad + spd_tw + int(5 * self.sf),
                 sy + self.font_speed - self.font_value),
                self.t['unit_s'], font=self._font(self.font_value),
                fill=self.text_dim, anchor='lt')

        # ── 分隔线 1 ──
        sep_y = sy + speed_area + int(2 * self.sf)
        d2.line([(x + pad, sep_y), (x + w - pad, sep_y)],
                fill=self.separator, width=1)

        # ── 回合统计 ──
        cy = sep_y + sep_space + section_gap
        d2.text((x + pad, cy), self.t['rally_label'],
                font=self._font(self.font_label),
                fill=accent, anchor='lt')
        cy += label_h

        # 距离 + 均速 同行
        dist = stats.get('rally_distance', 0)
        avg = stats.get('rally_avg_speed', 0)
        d2.text((x + pad, cy),
                f"{self.t['distance']} {dist:.2f}{self.t['unit_d']}",
                font=self._font(self.font_value),
                fill=self.text_white, anchor='lt')
        # 均速靠右
        avg_text = f"{self.t['avg']} {avg:.2f}"
        avg_font = self._font(self.font_value)
        avg_bbox = avg_font.getbbox(avg_text)
        avg_tw = avg_bbox[2] - avg_bbox[0]
        d2.text((x + w - pad - avg_tw, cy),
                avg_text, font=avg_font,
                fill=self.text_white, anchor='lt')
        cy += line_h

        # 极速
        mx = stats.get('rally_max_speed', 0)
        d2.text((x + pad, cy),
                f"{self.t['max']} {mx:.2f}{self.t['unit_s']}",
                font=self._font(self.font_value),
                fill=accent, anchor='lt')

        # ── 分隔线 2 ──
        sep2_y = cy + line_h + section_gap
        d2.line([(x + pad, sep2_y), (x + w - pad, sep2_y)],
                fill=self.separator, width=1)

        # ── 全场统计 (仅总距离) ──
        my = sep2_y + sep_space + section_gap
        d2.text((x + pad, my), self.t['match'],
                font=self._font(self.font_label),
                fill=accent, anchor='lt')
        my += label_h

        md = stats.get('match_distance', 0)
        d2.text((x + pad, my),
                f"{self.t['distance']} {md:.2f}{self.t['unit_d']}",
                font=self._font(self.font_value),
                fill=self.text_white, anchor='lt')

        return h
