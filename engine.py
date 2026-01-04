import time

from engine.pose_engine import PoseEngine
from engine.shot_segmenter import ShotSegmenter
from engine.speed_estimator import SpeedEstimator
from engine.coach import Coach
from engine.csv_logger import CSVLogger
from engine.highlight_generator import HighlightGenerator


class CricketEngine:
    """
    Central orchestrator engine
    """

    def __init__(self):
        self.pose_engine = PoseEngine()
        self.segmenter = ShotSegmenter()
        self.speed = SpeedEstimator()
        self.coach = Coach()
        self.logger = CSVLogger()

        self.video_name = "Rohit_Sharma"
        self.highlights = HighlightGenerator(
            output_dir=f"highlights/{self.video_name}"
        )

        self.frame_id = 0
        self.start_time = time.time()

    # --------------------------------------------------
    # MAIN PROCESS
    # --------------------------------------------------
    def process_frame(self, frame, frame_id=None):
        if frame_id is None:
            frame_id = self.frame_id

        highlight_path = None

        # 1️⃣ Pose (Uses new Tracking Logic)
        pose_data = self.pose_engine.process(frame)

        # 2️⃣ Shot segmentation
        shot_data = self.segmenter.update(pose_data, frame_id)

        # 3️⃣ Speed estimation
        speed_data = self.speed.update(pose_data)

        # 4️⃣ Coaching feedback
        coach_data = self.coach.evaluate(pose_data, speed_data)

        # 5️⃣ Highlight generation (ONLY when shot ends)
        if shot_data.get("ended"):
            highlight_path = self.highlights.save(
                shot_data["id"],
                shot_data["frames"],
                fps=pose_data.get("fps", 30)
            )

        # 6️⃣ CSV logging
        self.logger.write(
            frame_id=frame_id,
            timestamp=round(time.time() - self.start_time, 2),
            pose=pose_data,
            shot=shot_data,
            speed=speed_data,
            coach=coach_data
        )

        # 7️⃣ Data for UI
        ui_data = {
            "shot": shot_data.get("id", 0),
            "phase": shot_data.get("phase", "Idle"),
            "track_id": pose_data.get("track_id", "Search"),
            "weight": pose_data.get("weight", "Neutral"),
            **speed_data,
            **coach_data,
            "highlight": highlight_path
        }

        self.frame_id += 1
        return pose_data["annotated"], ui_data