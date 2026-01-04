import cv2
import numpy as np
from ultralytics import YOLO


class PoseEngine:
    """
    Handles:
    - Multi-player pose tracking (locks IDs across frames)
    - Visualization of IDs for easy manual selection
    - Target lock logic
    - Biomechanical analysis (Angles + Weight Transfer)
    """

    def __init__(self):
        # Using yolov8n-pose for speed; persist=True is key for tracking
        self.model = YOLO("yolov8n-pose.pt")
        self.target_id = None  # The ID we want to analyze specifically
        self.fps = 30 

    @staticmethod
    def calculate_angle(a, b, c):
        """Calculates angle between three points (a, b, c)."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        return int(angle)

    def calculate_weight_transfer(self, kpts):
        """
        Estimates weight distribution based on Hip x-position relative to Ankles.
        Returns: string (e.g., "60% Front", "Balanced")
        """
        try:
            # YOLOv8 Indices: 11: L Hip, 12: R Hip, 15: L Ankle, 16: R Ankle
            l_hip, r_hip = kpts[11][:2], kpts[12][:2]
            l_ankle, r_ankle = kpts[15][:2], kpts[16][:2]

            # Hip Midpoint (Approximation of Center of Mass)
            hip_center_x = (l_hip[0] + r_hip[0]) / 2
            
            feet_width = abs(l_ankle[0] - r_ankle[0])
            if feet_width < 10: return "Balanced"

            # Determine bounds
            min_x = min(l_ankle[0], r_ankle[0])
            max_x = max(l_ankle[0], r_ankle[0])
            
            # Position as a percentage (0 = Back Foot, 1 = Front Foot)
            pos_ratio = (hip_center_x - min_x) / (max_x - min_x)
            pos_ratio = np.clip(pos_ratio, 0, 1)

            # Map to text
            if pos_ratio > 0.6:
                return f"{int(pos_ratio * 100)}% Front"
            elif pos_ratio < 0.4:
                return f"{int((1 - pos_ratio) * 100)}% Back"
            else:
                return "Balanced"
        except:
            return "Neutral"

    def process(self, frame):
        """
        Process frame to track subjects and extract data for the target.
        """
        annotated = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        data = {
            "annotated": annotated,
            "gray": gray,
            "elbow": 0,
            "knee": 0,
            "weight": "Neutral",
            "wrist": None,
            "elbow_pt": None,
            "track_id": "Searching...",
            "fps": self.fps
        }

        # Enable tracking with persist=True to keep IDs consistent
        # NOTE: Requires 'pip install lapx'
        results = self.model.track(frame, persist=True, verbose=False)

        # Check if we have detections
        if (not results or results[0].boxes is None or 
            results[0].boxes.id is None or len(results[0].boxes) == 0):
            return data

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        kpts_all = results[0].keypoints.xy.cpu().numpy()

        # ---------------- 1. VISUALIZE TRACK IDs ----------------
        # Draw IDs on everyone so the user can see them
        for i, tid in enumerate(track_ids):
            x1, y1, x2, y2 = map(int, boxes[i])
            
            # Color coding: Green if it's our target, Orange otherwise
            color = (0, 255, 0) if (self.target_id is not None and tid == self.target_id) else (0, 165, 255)
            
            label = f"ID: {tid}"
            cv2.putText(
                annotated, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # ---------------- 2. SELECT TARGET ----------------
        target_idx = None
        current_track_id = None

        if self.target_id is not None:
            # User specified an ID, look for it
            if self.target_id in track_ids:
                target_idx = np.where(track_ids == self.target_id)[0][0]
                current_track_id = self.target_id
        else:
            # Default: Pick the largest person (likely the batsman)
            max_area = 0
            for i, box in enumerate(boxes):
                area = (box[2]-box[0]) * (box[3]-box[1])
                if area > max_area:
                    max_area = area
                    target_idx = i
                    current_track_id = track_ids[i]

        # ---------------- 3. EXTRACT METRICS ----------------
        if target_idx is not None:
            kpts = kpts_all[target_idx]
            try:
                # Calculate Biomechanics
                elbow_angle = self.calculate_angle(kpts[6][:2], kpts[8][:2], kpts[10][:2])
                knee_angle = self.calculate_angle(kpts[12][:2], kpts[14][:2], kpts[16][:2])
                weight_str = self.calculate_weight_transfer(kpts)

                wrist = tuple(kpts[10][:2].astype(int))
                elbow_pt = tuple(kpts[8][:2].astype(int))

                data.update({
                    "annotated": annotated, # Contains the ID drawings
                    "elbow": elbow_angle,
                    "knee": knee_angle,
                    "weight": weight_str,
                    "wrist": wrist,
                    "elbow_pt": elbow_pt,
                    "track_id": str(current_track_id)
                })
            except Exception:
                # Fallback if keypoints are obscured
                pass

        return data