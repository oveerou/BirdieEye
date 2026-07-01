"""Runtime camera drift correction via periodic court corner re-detection.

Concept:
  - Manual corners (from user annotation) are the ground-truth baseline.
  - Every ``check_interval`` frames, we run ``auto_detect_court_corners`` on
    the current frame to find where the court is *now*.
  - A homography H maps baseline corners → current corners.
  - To undo the drift, player centroids (in frame space) are transformed by
    H⁻¹ before being fed to CourtMapper.
  - If the homography's mean reprojection error exceeds ``max_error_px``, the
    correction is rejected (auto-detect was probably wrong on this frame).
"""

import cv2
import numpy as np

from .detector import auto_detect_court_corners


class CourtDriftCorrector:
    """Periodically re-detect court corners and correct camera drift."""

    def __init__(self, baseline_corners, check_interval=90, max_error_px=30.0):
        """
        Parameters
        ----------
        baseline_corners : list of 4 (x, y) tuples
            Manual corners in original-resolution frame space.
        check_interval : int
            Run auto-detect every *check_interval* court frames (~seconds @ 30fps).
        max_error_px : float
            Reject homography if mean reprojection error exceeds this.
        """
        self.baseline = np.array(baseline_corners, dtype=np.float32)
        self.check_interval = check_interval
        self.max_error_px = max_error_px

        # Current correction state
        self._H = None          # baseline → current
        self._H_inv = None      # current → baseline (the one we apply)
        self.frames_since_check = 0
        self.n_corrections = 0
        self.n_rejected = 0
        self.last_error_px = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_check(self, frame):
        """Run auto-detection if enough frames have elapsed.

        Returns the mean reprojection error (px) if a correction was computed,
        or *None* if no correction was applied.
        """
        self.frames_since_check += 1
        if self.frames_since_check < self.check_interval:
            return None

        self.frames_since_check = 0
        corners, _mask, _debug = auto_detect_court_corners(frame)
        if corners is None or len(corners) != 4:
            return None

        detected = np.array(corners, dtype=np.float32)

        H, _ = cv2.findHomography(self.baseline.reshape(-1, 1, 2),
                                   detected.reshape(-1, 1, 2))
        if H is None:
            return None

        # Mean reprojection error
        projected = cv2.perspectiveTransform(
            self.baseline.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        error = float(np.mean(np.sqrt(np.sum((projected - detected) ** 2, axis=1))))
        self.last_error_px = error

        if error > self.max_error_px:
            self.n_rejected += 1
            try:
                print(f"[drift] homography rejected: error={error:.1f}px "
                      f"(threshold={self.max_error_px}px)")
            except OSError:
                pass
            return None

        self._H = H
        self._H_inv = cv2.invert(H)[1]
        self.n_corrections += 1
        try:
            print(f"[drift] correction updated: error={error:.1f}px, "
                  f"total corrections={self.n_corrections}")
        except OSError:
            pass
        return error

    def correct(self, points):
        """Apply the current drift correction to a list of (x, y) points.

        Returns corrected points in the same format.  If no correction is
        available (first check hasn't run yet, or last check was rejected),
        the original points are returned unchanged.
        """
        if self._H_inv is None:
            return points

        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        corrected = cv2.perspectiveTransform(pts, self._H_inv).reshape(-1, 2)
        return [tuple(p) for p in corrected]

    @property
    def is_active(self):
        """Whether a valid correction matrix is currently available."""
        return self._H_inv is not None
