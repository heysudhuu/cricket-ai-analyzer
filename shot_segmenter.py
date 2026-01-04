from collections import deque


class ShotSegmenter:
    """
    Responsible for:
    - Detecting shot phases
    - Segmenting shots
    - Buffering frames for highlights

    Stateless with respect to pose detection.
    """

    def __init__(self, buffer_size=120):
        """
        buffer_size: max frames stored per shot
        """
        self.current_shot = None
        self.shot_id = 0
        self.frame_buffer = deque(maxlen=buffer_size)

    # --------------------------------------------------
    # Phase logic (rule-based)
    # --------------------------------------------------
    @staticmethod
    def detect_phase(elbow_angle):
        if elbow_angle > 150:
            return "Backlift"
        elif elbow_angle > 120:
            return "Downswing"
        elif elbow_angle > 90:
            return "Contact"
        else:
            return "FollowThrough"

    # --------------------------------------------------
    # Main update
    # --------------------------------------------------
    def update(self, pose_data, frame_id):
        """
        Input:
            pose_data: dict from PoseEngine
            frame_id: current frame index

        Output:
            dict:
            {
              "id": int,
              "phase": str,
              "started": bool,
              "ended": bool,
              "frames": list (only when ended)
            }
        """

        elbow = pose_data.get("elbow", 0)
        frame = pose_data.get("annotated")

        phase = self.detect_phase(elbow)

        # Default return
        result = {
            "id": self.shot_id,
            "phase": phase,
            "started": False,
            "ended": False,
            "frames": None
        }

        # ---------------- Start Shot ----------------
        if phase == "Backlift" and self.current_shot is None:
            self.shot_id += 1
            self.current_shot = {
                "id": self.shot_id,
                "start_frame": frame_id
            }
            self.frame_buffer.clear()
            result["started"] = True

        # ---------------- Buffer Frames ----------------
        if self.current_shot is not None:
            self.frame_buffer.append(frame.copy())

        # ---------------- End Shot ----------------
        if phase == "FollowThrough" and self.current_shot is not None:
            result.update({
                "id": self.current_shot["id"],
                "ended": True,
                "frames": list(self.frame_buffer)
            })

            self.current_shot = None
            self.frame_buffer.clear()

        # Active shot ID
        if self.current_shot:
            result["id"] = self.current_shot["id"]

        return result
