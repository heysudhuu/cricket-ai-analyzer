import cv2
import numpy as np
from collections import deque


class SpeedEstimator:
    """
    - Smooth bat speed using EMA
    - Real ball speed using motion tracking
    """

    def __init__(self, fps=30):
        self.fps = fps

        # Approx calibration (can refine later)
        self.meters_per_pixel = 0.0025

        # ---------------- BAT SPEED ----------------
        self.prev_wrist = None
        self.bat_ema = None
        self.alpha = 0.3  # smoothing factor

        # ---------------- BALL SPEED ----------------
        self.prev_gray = None
        self.prev_ball_center = None
        self.ball_speeds = deque(maxlen=5)

    # --------------------------------------------------
    # BAT SPEED (wrist-based, smoothed)
    # --------------------------------------------------
    def _bat_speed(self, wrist):
        if wrist is None or self.prev_wrist is None:
            self.prev_wrist = wrist
            return 0.0

        px_dist = np.linalg.norm(
            np.array(wrist) - np.array(self.prev_wrist)
        )
        self.prev_wrist = wrist

        raw_speed = px_dist * self.meters_per_pixel * self.fps * 3.6

        # EMA smoothing
        if self.bat_ema is None:
            self.bat_ema = raw_speed
        else:
            self.bat_ema = (
                self.alpha * raw_speed
                + (1 - self.alpha) * self.bat_ema
            )

        # realistic cap
        return round(min(self.bat_ema, 160.0), 2)

    # --------------------------------------------------
    # BALL SPEED (motion-based)
    # --------------------------------------------------
    def _ball_speed(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        # Frame differencing
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.medianBlur(thresh, 5)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0

        # Choose small fast-moving object
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        # Filter noise (cricket ball size range)
        if area < 5 or area > 300:
            return 0.0

        x, y, w, h = cv2.boundingRect(c)
        center = (x + w // 2, y + h // 2)

        if self.prev_ball_center is None:
            self.prev_ball_center = center
            return 0.0

        px_dist = np.linalg.norm(
            np.array(center) - np.array(self.prev_ball_center)
        )
        self.prev_ball_center = center

        speed = px_dist * self.meters_per_pixel * self.fps * 3.6
        self.ball_speeds.append(speed)

        return round(min(np.mean(self.ball_speeds), 160.0), 2)

    # --------------------------------------------------
    # MAIN UPDATE (called from engine)
    # --------------------------------------------------
    def update(self, pose_data):
        bat_speed = self._bat_speed(pose_data.get("wrist"))
        ball_speed = self._ball_speed(pose_data.get("gray"))

        # Arm speed approx ratio (biomechanics heuristic)
        arm_speed = round(bat_speed * 0.6, 2)

        return {
            "bat_speed": bat_speed,
            "ball_speed": ball_speed,
            "arm_speed": arm_speed
        }
