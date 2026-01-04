import csv
import os
from datetime import datetime


class CSVLogger:
    """
    Handles structured analytics logging.

    Writes one row per frame with:
    - Shot metadata
    - Biomechanics
    - Speeds
    - Coaching feedback

    Research / report ready.
    """

    def __init__(self, filename="cricket_analysis.csv"):
        self.filename = filename
        self._init_file()

    def _init_file(self):
        file_exists = os.path.exists(self.filename)

        self.file = open(self.filename, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        if not file_exists:
            self.writer.writerow([
                "Timestamp",
                "FrameID",
                "ShotID",
                "Phase",
                "ElbowAngle(deg)",
                "KneeAngle(deg)",
                "BatSpeed(km/h)",
                "BallSpeed(km/h)",
                "ArmSpeed(km/h)",
                "CoachingFeedback"
            ])

    def write(self, frame_id, timestamp, pose, shot, speed, coach):
        """
        Inputs:
            frame_id: int
            timestamp: float (seconds)
            pose: dict from PoseEngine
            shot: dict from ShotSegmenter
            speed: dict from SpeedEstimator
            coach: dict from Coach
        """

        row = [
            datetime.now().isoformat(timespec="seconds"),
            frame_id,
            shot.get("id", 0),
            shot.get("phase", "Idle"),
            pose.get("elbow", 0),
            pose.get("knee", 0),
            speed.get("bat_speed", 0),
            speed.get("ball_speed", 0),
            speed.get("arm_speed", 0),
            coach.get("feedback", "")
        ]

        self.writer.writerow(row)
        self.file.flush()  # ensure safety on crashes

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass
