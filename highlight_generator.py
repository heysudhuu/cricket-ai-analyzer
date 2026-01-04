class HighlightGenerator:
    def __init__(self, output_dir="highlights"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
def save(self, shot_id, frames, fps=30):
    if not frames or len(frames) < 5:
        return None

    h, w, _ = frames[0].shape

    filename = f"shot_{shot_id}.mp4"
    path = os.path.join(self.output_dir, filename)

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    for frame in frames:
        writer.write(frame)

    writer.release()
    return path
