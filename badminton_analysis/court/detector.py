import cv2
import numpy as np


def _line_angle(x1, y1, x2, y2):
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return angle % 180


def _line_intersection(line_a, line_b):
    x1, y1, x2, y2 = line_a
    x3, y3, x4, y4 = line_b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return np.array([px, py], dtype=np.float32)


def _inside_image(point, width, height, margin=0):
    return margin <= point[0] <= width - 1 - margin and margin <= point[1] <= height - 1 - margin


def _polygon_area(points):
    return abs(cv2.contourArea(np.array(points, dtype=np.float32)))


def _is_convex_quad(points):
    pts = np.array(points, dtype=np.float32)
    return len(cv2.convexHull(pts)) == 4 and cv2.isContourConvex(pts.astype(np.int32))


def _project_court_points(matrix, points):
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def _badminton_line_alignment_score(corners, line_mask):
    height, width = line_mask.shape[:2]
    image_points = np.array(corners, dtype=np.float32)
    court_points = np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(court_points, image_points)

    expected = np.zeros((height, width), dtype=np.uint8)
    horizontal_ys = [0, 0.76, 4.72, 6.7, 8.68, 12.64, 13.4]
    vertical_xs = [0, 0.46, 3.05, 5.64, 6.1]

    for y in horizontal_ys:
        p1, p2 = _project_court_points(matrix, [(0, y), (6.1, y)])
        cv2.line(expected, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), 255, 5)

    for x in vertical_xs:
        p1, p2 = _project_court_points(matrix, [(x, 0), (x, 13.4)])
        cv2.line(expected, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), 255, 5)

    center_segments = [((3.05, 0), (3.05, 4.72)), ((3.05, 8.68), (3.05, 13.4))]
    for p1_court, p2_court in center_segments:
        p1, p2 = _project_court_points(matrix, [p1_court, p2_court])
        cv2.line(expected, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), 255, 5)

    expected_pixels = cv2.countNonZero(expected)
    if expected_pixels == 0:
        return 0.0

    dilated_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
    overlap = cv2.bitwise_and(expected, dilated_mask)
    return cv2.countNonZero(overlap) / expected_pixels


def _badminton_horizontal_pattern_score(corners, horizontal_lines, image_shape):
    if not horizontal_lines:
        return 0.0

    height, _width = image_shape[:2]
    image_points = np.array(corners, dtype=np.float32)
    court_points = np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(court_points, image_points)

    expected_court_y = [0, 0.76, 4.72, 6.7, 8.68, 12.64, 13.4]
    weights = [5.0, 4.0, 1.6, 0.8, 1.6, 4.0, 5.0]
    detected_y = [float(line["mid"][1]) for line in horizontal_lines]
    tolerance = max(12.0, height * 0.022)

    weighted_score = 0.0
    total_weight = 0.0
    for court_y, weight in zip(expected_court_y, weights):
        p1, p2 = _project_court_points(matrix, [(0, court_y), (6.1, court_y)])
        expected_y = float((p1[1] + p2[1]) / 2.0)
        nearest = min(abs(y - expected_y) for y in detected_y)
        weighted_score += max(0.0, 1.0 - nearest / tolerance) * weight
        total_weight += weight

    return weighted_score / total_weight


def _horizontal_segment_support(line, p_left, p_right, image_width):
    x1, _y1, x2, _y2 = line["points"]
    seg_min, seg_max = sorted((x1, x2))
    quad_min, quad_max = sorted((float(p_left[0]), float(p_right[0])))
    overshoot = max(0.0, seg_min - quad_min) + max(0.0, quad_max - seg_max)
    return max(0.0, 1.0 - overshoot / max(1.0, image_width * 0.20))


def _side_segment_support(line, p_top, p_bottom, image_height):
    _x1, y1, _x2, y2 = line["points"]
    seg_min, seg_max = sorted((y1, y2))
    quad_min, quad_max = sorted((float(p_top[1]), float(p_bottom[1])))
    overshoot = max(0.0, seg_min - quad_min) + max(0.0, quad_max - seg_max)
    return max(0.0, 1.0 - overshoot / max(1.0, image_height * 0.16))


def _promote_far_baseline(lines, horizontal_lines, image_shape):
    height, _width = image_shape[:2]
    top_line, bottom_line, left_line, right_line = lines
    top_y = float(top_line["mid"][1])
    candidates = []

    for line in horizontal_lines:
        candidate_y = float(line["mid"][1])
        gap = top_y - candidate_y
        if gap <= height * 0.04 or gap >= height * 0.09:
            continue
        if candidate_y < height * 0.28:
            continue
        if line["length"] < top_line["length"] * 0.65:
            continue
        next_gap = min(
            [float(other["mid"][1]) - top_y for other in horizontal_lines if float(other["mid"][1]) > top_y] or [height]
        )
        if gap > next_gap * 0.9:
            continue
        candidates.append(line)

    if not candidates:
        return lines

    promoted_top = max(candidates, key=lambda item: item["length"])
    return promoted_top, bottom_line, left_line, right_line


def _dedupe_lines(lines, orientation, max_count):
    selected = []
    for line in sorted(lines, key=lambda item: item["length"], reverse=True):
        duplicate = False
        for existing in selected:
            if orientation == "horizontal":
                same_band = abs(line["mid"][1] - existing["mid"][1]) < 18
                same_angle = abs(line["angle"] - existing["angle"]) < 8
            else:
                same_band = abs(line["mid"][0] - existing["mid"][0]) < 22
                same_angle = abs(line["angle"] - existing["angle"]) < 10
            if same_band and same_angle:
                duplicate = True
                break
        if not duplicate:
            selected.append(line)
        if len(selected) >= max_count:
            break
    return selected


def build_court_line_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    green_mask = (((h >= 35) & (h <= 95) & (s >= 30) & (v >= 45)).astype(np.uint8)) * 255
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=2)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(green_mask)
    if count > 1:
        image_area = image.shape[0] * image.shape[1]
        candidates = [idx for idx in range(1, count) if stats[idx, cv2.CC_STAT_AREA] > image_area * 0.08]
        if candidates:
            court_idx = max(candidates, key=lambda idx: stats[idx, cv2.CC_STAT_AREA])
            green_mask = ((labels == court_idx).astype(np.uint8)) * 255

    court_roi = cv2.dilate(green_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)), iterations=1)
    white_mask = (((s <= 95) & (v >= 135)).astype(np.uint8)) * 255
    white_mask = cv2.bitwise_and(white_mask, court_roi)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 45, 135)
    edges = cv2.bitwise_and(edges, court_roi)

    merged = cv2.bitwise_or(white_mask, edges)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)), iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return merged


def detect_court_line_segments(image):
    height, width = image.shape[:2]
    mask = build_court_line_mask(image)
    min_line_length = max(46, int(min(width, height) * 0.10))
    max_gap = max(14, int(min(width, height) * 0.045))
    raw_lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=min_line_length,
        maxLineGap=max_gap,
    )
    if raw_lines is None:
        return [], [], mask

    edge_margin = max(10, int(min(width, height) * 0.02))
    horizontal = []
    side = []
    for raw_line in raw_lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in raw_line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_line_length:
            continue

        if (
            min(x1, x2) <= edge_margin
            or max(x1, x2) >= width - 1 - edge_margin
            or min(y1, y2) <= edge_margin
            or max(y1, y2) >= height - 1 - edge_margin
        ):
            continue

        angle = _line_angle(x1, y1, x2, y2)
        segment = {
            "points": (x1, y1, x2, y2),
            "length": length,
            "mid": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            "angle": angle,
        }
        if min(angle, 180 - angle) <= 16:
            horizontal.append(segment)
        elif 45 <= angle <= 135:
            side.append(segment)

    return _dedupe_lines(horizontal, "horizontal", 14), _dedupe_lines(side, "side", 18), mask


def _score_court_quad(corners, image_shape, lines, line_mask, horizontal_lines):
    height, width = image_shape[:2]
    top_left, top_right, bottom_right, bottom_left = corners
    points = np.array(corners, dtype=np.float32)

    if not all(_inside_image(point, width, height, margin=8) for point in points):
        return None
    if not _is_convex_quad(points):
        return None

    area = _polygon_area(points)
    image_area = width * height
    if area < image_area * 0.12:
        return None

    top_width = float(np.linalg.norm(top_right - top_left))
    bottom_width = float(np.linalg.norm(bottom_right - bottom_left))
    left_height = float(np.linalg.norm(bottom_left - top_left))
    right_height = float(np.linalg.norm(bottom_right - top_right))
    avg_height = (left_height + right_height) / 2.0
    if min(top_width, bottom_width, avg_height) <= 1:
        return None
    if avg_height < height * 0.34:
        return None
    if bottom_width < top_width * 0.82:
        return None

    center = np.mean(points, axis=0)
    top_y = (top_left[1] + top_right[1]) / 2.0
    bottom_y = (bottom_left[1] + bottom_right[1]) / 2.0
    width_ratio = top_width / bottom_width
    if not (width * 0.42 <= center[0] <= width * 0.76 and height * 0.58 <= center[1] <= height * 0.86):
        return None
    if top_y < height * 0.30 or bottom_y < height * 0.74:
        return None
    if not (0.45 <= width_ratio <= 0.86):
        return None

    min_edge_distance = min(
        np.min(points[:, 0]),
        width - 1 - np.max(points[:, 0]),
        np.min(points[:, 1]),
        height - 1 - np.max(points[:, 1]),
    )
    edge_limit = max(10, min(width, height) * 0.02)
    if min_edge_distance < edge_limit:
        return None

    left_dx = bottom_left[0] - top_left[0]
    right_dx = bottom_right[0] - top_right[0]
    if left_dx > width * 0.16 or right_dx < -width * 0.16:
        return None

    top_line, bottom_line, left_line, right_line = lines
    area_score = min(area / (image_area * 0.42), 1.0) * 44
    height_score = min(avg_height / (height * 0.62), 1.0) * 22
    perspective_score = (1.0 - min(abs(width_ratio - 0.66) / 0.22, 1.0)) * 16
    center_x_score = (1.0 - min(abs(center[0] - width * 0.50) / (width * 0.24), 1.0)) * 18
    center_y_score = (1.0 - min(abs(center[1] - height * 0.70) / (height * 0.20), 1.0)) * 8
    bottom_score = min(max((bottom_y / height - 0.74) / 0.18, 0), 1.0) * 8
    edge_score = min(min_edge_distance / (min(width, height) * 0.08), 1.0) * 6
    line_score = min(
        (top_line["length"] + bottom_line["length"] + left_line["length"] + right_line["length"])
        / (width * 2.0 + height),
        1.0,
    ) * 6
    horizontal_support = (
        _horizontal_segment_support(top_line, top_left, top_right, width)
        + _horizontal_segment_support(bottom_line, bottom_left, bottom_right, width)
    ) / 2.0
    side_support = (
        _side_segment_support(left_line, top_left, bottom_left, height)
        + _side_segment_support(right_line, top_right, bottom_right, height)
    ) / 2.0
    support_score = (horizontal_support * 0.55 + side_support * 0.45) * 24
    alignment_score = _badminton_line_alignment_score(corners, line_mask) * 10
    pattern_score = _badminton_horizontal_pattern_score(corners, horizontal_lines, image_shape) * 110
    return area_score + height_score + perspective_score + center_x_score + center_y_score + bottom_score + edge_score + line_score + support_score + alignment_score + pattern_score


def _order_corners_tl_tr_br_bl(pts):
    """Order 4 points as TL, TR, BR, BL (top-left, top-right, bottom-right, bottom-left)."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)               # x + y
    d = pts[:, 0] - pts[:, 1]         # x - y
    tl = pts[np.argmin(s)]            # smallest x+y -> top-left
    br = pts[np.argmax(s)]            # largest x+y -> bottom-right
    tr = pts[np.argmax(d)]            # largest x-y -> top-right
    bl = pts[np.argmin(d)]            # smallest x-y -> bottom-left
    return [tuple(v.astype(int)) for v in (tl, tr, br, bl)]


def _quad_aspect(corners):
    """Return width/height ratio (>=1) of an ordered (TL,TR,BR,BL) quad.

    Uses the axis-aligned bounding box so perspective doesn't skew the ratio.
    """
    if len(corners) != 4:
        return 0.0
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width <= 0 or height <= 0:
        return 0.0
    return max(width, height) / min(width, height)


# Court is 6.1 x 13.4 m, aspect 2.2. Allow perspective tilt range.
COURT_ASPECT_MIN = 1.4
COURT_ASPECT_MAX = 4.5


def detect_court_by_color(image, color: str = "red"):
    """Color-threshold based court detector. Fast and reliable for BWF red/green courts.

    Returns 4 corners ordered TL, TR, BR, BL, or None if no suitable region is found.
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color == "red":
        # Red wraps around in HSV (hue 0 and 180 are both red).
        lower1 = np.array([0, 80, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 80, 50])
        upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    elif color == "green":
        lower = np.array([35, 60, 50])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        return None

    # Morphological cleanup: close holes, then drop small specks
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    # A real BWF court fills >= ~30% of the frame; logos/grass outside the
    # court are usually smaller. Reject them to avoid false positives.
    if area < 0.30 * h * w:
        return None
    hull = cv2.convexHull(largest)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
    if len(approx) == 4:
        pts = approx.reshape(-1, 2)
    else:
        rect = cv2.minAreaRect(hull)
        pts = cv2.boxPoints(rect)
    corners = _order_corners_tl_tr_br_bl(pts)
    # Reject shapes that don't look like a badminton court aspect ratio.
    aspect = _quad_aspect(corners)
    if not (COURT_ASPECT_MIN <= aspect <= COURT_ASPECT_MAX):
        return None
    return corners


def auto_detect_court_corners(image, preferred_color: str = "red"):
    # Fast path: color-threshold for BWF red/green courts.
    for color in (preferred_color, "red" if preferred_color != "red" else "green"):
        if color is None:
            continue
        corners = detect_court_by_color(image, color=color)
        if corners is not None:
            debug = {"method": "color", "color": color}
            return corners, None, debug

    # Fallback: Hough-line based detection (works on any surface).
    horizontal_lines, side_lines, mask = detect_court_line_segments(image)
    debug = {"method": "line", "horizontal": horizontal_lines, "side": side_lines, "score": None}
    if len(horizontal_lines) < 2 or len(side_lines) < 2:
        return None, mask, debug

    height, width = image.shape[:2]
    best = None
    horizontals = sorted(horizontal_lines, key=lambda item: item["mid"][1])
    sides = sorted(side_lines, key=lambda item: item["mid"][0])

    for top_idx, top_line in enumerate(horizontals):
        for bottom_line in horizontals[top_idx + 1:]:
            if bottom_line["mid"][1] - top_line["mid"][1] < height * 0.28:
                continue
            for left_idx, left_line in enumerate(sides):
                for right_line in sides[left_idx + 1:]:
                    if right_line["mid"][0] - left_line["mid"][0] < width * 0.22:
                        continue

                    intersections = [
                        _line_intersection(top_line["points"], left_line["points"]),
                        _line_intersection(top_line["points"], right_line["points"]),
                        _line_intersection(bottom_line["points"], right_line["points"]),
                        _line_intersection(bottom_line["points"], left_line["points"]),
                    ]
                    if any(point is None for point in intersections):
                        continue

                    score = _score_court_quad(intersections, image.shape, (top_line, bottom_line, left_line, right_line), mask, horizontal_lines)
                    if score is None:
                        continue
                    if best is None or score > best["score"]:
                        best = {"corners": intersections, "score": score, "lines": (top_line, bottom_line, left_line, right_line)}

    if best is None:
        return None, mask, debug

    selected_lines = _promote_far_baseline(best["lines"], horizontal_lines, image.shape)
    top_line, bottom_line, left_line, right_line = selected_lines
    corners_float = [
        _line_intersection(top_line["points"], left_line["points"]),
        _line_intersection(top_line["points"], right_line["points"]),
        _line_intersection(bottom_line["points"], right_line["points"]),
        _line_intersection(bottom_line["points"], left_line["points"]),
    ]
    if any(point is None for point in corners_float):
        corners_float = best["corners"]

    debug["score"] = best["score"]
    corners = [(int(round(point[0])), int(round(point[1]))) for point in corners_float]
    return corners, mask, debug


def render_auto_court_preview(image, corners, roi_corners=None, debug=None):
    preview = image.copy()
    if debug:
        for line in debug.get("horizontal", []):
            cv2.line(preview, line["points"][:2], line["points"][2:], (0, 220, 255), 2)
        for line in debug.get("side", []):
            cv2.line(preview, line["points"][:2], line["points"][2:], (255, 180, 0), 2)

    if corners:
        points = np.array(corners, dtype=np.int32)
        cv2.polylines(preview, [points], True, (0, 255, 0), 3)
        for idx, point in enumerate(corners, start=1):
            cv2.circle(preview, point, 6, (0, 0, 255), -1)
            cv2.putText(preview, str(idx), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    if roi_corners:
        cv2.rectangle(preview, roi_corners[0], roi_corners[1], (255, 0, 0), 3)

    cv2.rectangle(preview, (0, 0), (preview.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(
        preview,
        "Auto court detection: Enter/Y accept, M/R/Esc manual",
        (16, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview
