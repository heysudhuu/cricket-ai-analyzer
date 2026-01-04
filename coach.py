class Coach:
    """
    Real-time coaching feedback engine.

    Uses biomechanical rules to generate
    explainable, human-readable feedback.
    """

    def __init__(self):
        # Thresholds (can be tuned later)
        self.MIN_BAT_SPEED = 60      # km/h
        self.MIN_ELBOW_ANGLE = 120   # degrees
        self.MIN_KNEE_ANGLE = 140    # degrees

    def evaluate(self, pose_data, speed_data):
        """
        Inputs:
            pose_data: dict from PoseEngine
            speed_data: dict from SpeedEstimator

        Returns:
            {
              "feedback": str
            }
        """

        elbow = pose_data.get("elbow", 0)
        knee = pose_data.get("knee", 0)

        bat_speed = speed_data.get("bat_speed", 0)
        arm_speed = speed_data.get("arm_speed", 0)
        ball_speed = speed_data.get("ball_speed", 0)

        # ---------------- PRIORITY RULES ----------------
        # Order matters: most critical feedback first

        if bat_speed and bat_speed < self.MIN_BAT_SPEED:
            return {
                "feedback": "Increase bat swing speed for better power"
            }

        if elbow and elbow < self.MIN_ELBOW_ANGLE:
            return {
                "feedback": "Start downswing earlier – elbow collapsing"
            }

        if knee and knee < self.MIN_KNEE_ANGLE:
            return {
                "feedback": "Bend front knee more for balance"
            }

        if arm_speed and arm_speed < bat_speed * 0.6:
            return {
                "feedback": "Accelerate forearm more through contact"
            }

        if ball_speed and ball_speed < bat_speed * 0.7:
            return {
                "feedback": "Late contact – meet the ball earlier"
            }

        # ---------------- POSITIVE FEEDBACK ----------------
        if bat_speed >= 80 and elbow >= 140 and knee >= 150:
            return {
                "feedback": "Excellent shot mechanics 👍"
            }

        # ---------------- DEFAULT ----------------
        return {
            "feedback": "Good control – keep consistency"
        }
