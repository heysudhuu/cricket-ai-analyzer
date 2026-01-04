
import sys
import os
import cv2
import traceback
from functools import partial

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QMessageBox, QSlider,
    QFrame, QLineEdit, QGridLayout, QProgressBar, QListWidget,
    QListWidgetItem, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence, QIcon

# Try import CricketEngine. If this fails, show a friendly error later.
try:
    from engine.engine import CricketEngine
except Exception as e:
    CricketEngine = None
    import_error = e


# ---------------------------
# VideoWorker (background)
# ---------------------------
class VideoWorker(QThread):
    frame_signal = pyqtSignal(QImage)        # updated annotated frame
    metric_signal = pyqtSignal(dict)         # updated metrics + frame_id
    finished_signal = pyqtSignal()           # finished processing
    error_signal = pyqtSignal(str)           # error message

    def __init__(self, video_path, fps=30, target_id=None):
        super().__init__()
        self.video_path = video_path
        self.running = False
        self.paused = False
        self._target_id = target_id
        self._fps = fps

        # lazy init engine in thread to avoid issues on main thread
        self.engine = None
        self.cap = None

    def run(self):
        try:
            if CricketEngine is None:
                raise RuntimeError(f"Failed to import CricketEngine: {import_error}")

            self.engine = CricketEngine()
            # set target id if provided
            if self._target_id is not None:
                try:
                    self.engine.pose_engine.target_id = int(self._target_id)
                except Exception:
                    self.engine.pose_engine.target_id = self._target_id

            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open video file.")

            self.running = True
            frame_id = 0
            frame_delay_ms = int(max(1, round(1000 / max(1, self._fps))))

            while self.running:
                if self.paused:
                    self.msleep(50)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    break

                try:
                    annotated, data = self.engine.process_frame(frame, frame_id)
                except Exception as e:
                    # forward engine errors but continue
                    tb = traceback.format_exc()
                    self.error_signal.emit(f"Engine error: {e}\n{tb}")
                    annotated = frame.copy()
                    data = {
                        "bat_speed": 0,
                        "arm_speed": 0,
                        "elbow": 0,
                        "knee": 0,
                        "weight": "Neutral",
                        "track_id": "N/A",
                        "feedback": "Processing error",
                        "frame_id": frame_id
                    }

                # convert BGR->RGB and pack into QImage
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # include frame_id in data for syncing UI/timeline
                data["frame_id"] = frame_id

                self.frame_signal.emit(qt_img)
                self.metric_signal.emit(data)

                frame_id += 1
                self.msleep(frame_delay_ms)

        except Exception as e:
            tb = traceback.format_exc()
            self.error_signal.emit(f"Worker error: {e}\n{tb}")
        finally:
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
            self.running = False
            self.finished_signal.emit()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        # ensure resume so loop can break if currently paused
        self.paused = False
        self.wait()

    def seek(self, frame_no):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_no)))


# ---------------------------
# Helper functions
# ---------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def qpixmap_from_image_path(path, max_size=(320, 180)):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pix = QPixmap.fromImage(img)
    return pix.scaled(*max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ---------------------------
# Main App Window
# ---------------------------
class CricketAIApp(QWidget):
    def __init__(self):
        super().__init__()

        # if engine import previously failed, notify
        self.import_error = None
        if CricketEngine is None:
            self.import_error = import_error

        self.worker = None
        self.video_path = None
        self.highlights_dir = os.path.join(os.getcwd(), "highlights", "Rohit_Sharma")
        ensure_dir(self.highlights_dir)
        self.csv_path = "cricket_analysis.csv"  # default from CSVLogger

        self.init_ui()
        self.install_shortcuts()

    def init_ui(self):
        self.setWindowTitle("Cricket AI Analyzer — Sudhanshu Yadav")
        self.setMinimumSize(1280, 880)

        # Dark theme stylesheet
        self.setStyleSheet("""
            QWidget { background-color: #0e1720; color: #e6eef6; font-family: "Segoe UI", Roboto, Arial; }
            QLabel#title { font-size: 20px; font-weight: 700; color: #eaf2ff; }
            QPushButton { background-color: #1f6feb; border-radius: 8px; padding: 8px 12px; color: white; }
            QPushButton:hover { background-color: #2b84ff; }
            QPushButton:disabled { background-color: #444e5a; color: #aab6c8; }
            QFrame.card { background-color: #0f2330; border-radius: 10px; padding: 12px; }
            QLabel.metric { font-size: 28px; font-weight: 700; color: #dff3ff; }
            QLabel.small { color: #9fb4c8; }
            QTextEdit { background-color: #071021; color: #dbeefc; border-radius: 8px; padding: 8px; }
            QSlider::groove:horizontal { height: 8px; background: #122733; border-radius: 4px; }
            QSlider::handle:horizontal { background: #1f6feb; width: 14px; border-radius: 7px; margin: -3px 0; }
            QListWidget { background-color: #071021; border-radius: 8px; }
            QLineEdit { background-color: #071021; border-radius: 6px; padding: 6px; color: #e6eef6; }
            QSpinBox { background-color: #071021; border-radius: 6px; padding: 6px; color: #e6eef6; }
        """)

        # Top bar
        top_bar = QHBoxLayout()
        title = QLabel("Cricket AI Studio Pro")
        title.setObjectName("title")
        subtitle = QLabel("— Enhanced UI/UX")
        subtitle.setStyleSheet("color: #9fb4c8; margin-left: 8px;")
        top_bar.addWidget(title)
        top_bar.addWidget(subtitle)
        top_bar.addStretch()

        status_label = QLabel("Status: Idle")
        status_label.setObjectName("status_label")
        self.status_label = status_label
        top_bar.addWidget(status_label)

        # Layout main
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_bar)

        content = QHBoxLayout()
        main_layout.addLayout(content)

        # --- Left Controls ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)

        # control buttons
        controls_frame = QFrame()
        controls_frame.setObjectName("controls_frame")
        controls_frame.setProperty("class", "card")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setSpacing(8)

        self.btn_load = QPushButton("📂 Load Video (L)")
        self.btn_start = QPushButton("▶ Start")
        self.btn_pause = QPushButton("⏸ Pause (Space)")
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_open_highlights = QPushButton("📁 Open Highlights (H)")
        self.btn_export_csv = QPushButton("📤 Export CSV")

        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(False)

        controls_layout.addWidget(self.btn_load)
        row = QHBoxLayout()
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_pause)
        row.addWidget(self.btn_stop)
        controls_layout.addLayout(row)
        controls_layout.addWidget(self.btn_open_highlights)
        controls_layout.addWidget(self.btn_export_csv)

        # options
        opts_frame = QFrame()
        opts_frame.setProperty("class", "card")
        opts_layout = QVBoxLayout(opts_frame)
        opts_layout.setSpacing(6)
        opts_layout.addWidget(QLabel("Options"))

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setValue(30)
        fps_row.addWidget(self.fps_spin)
        fps_row.addStretch()
        opts_layout.addLayout(fps_row)

        tid_row = QHBoxLayout()
        tid_row.addWidget(QLabel("Target ID:"))
        self.target_input = QLineEdit()
        self.target_input.setPlaceholderText("leave blank => auto")
        tid_row.addWidget(self.target_input)
        opts_layout.addLayout(tid_row)

        self.chk_coach = QPushButton("Toggle Coach: ON")
        self.chk_coach.setCheckable(True)
        self.chk_coach.setChecked(True)
        opts_layout.addWidget(self.chk_coach)

        left_panel.addWidget(controls_frame)
        left_panel.addWidget(opts_frame)
        left_panel.addStretch()

        content.addLayout(left_panel, 1)

        # --- Center: Video + Timeline + Highlights ---
        center_panel = QVBoxLayout()
        center_panel.setSpacing(10)

        # Video display card
        video_frame = QFrame()
        video_frame.setProperty("class", "card")
        video_layout = QVBoxLayout(video_frame)

        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(860, 520)
        self.video_label.setStyleSheet("border-radius: 8px; background-color: #071421;")
        video_layout.addWidget(self.video_label)

        # overlay small metrics (will be updated)
        overlay_row = QHBoxLayout()
        self.overlay_bat = QLabel("Bat: -- km/h")
        self.overlay_arm = QLabel("Arm: -- km/h")
        self.overlay_track = QLabel("TrackID: --")
        for w in (self.overlay_bat, self.overlay_arm, self.overlay_track):
            w.setStyleSheet("color: #c7e9ff; font-weight: 600;")
            overlay_row.addWidget(w)
        overlay_row.addStretch()
        video_layout.addLayout(overlay_row)

        # timeline
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setMinimum(0)
        self.timeline.setValue(0)
        video_layout.addWidget(self.timeline)

        center_panel.addWidget(video_frame)

        # highlights thumbnails
        thumbs_frame = QFrame()
        thumbs_frame.setProperty("class", "card")
        thumbs_layout = QVBoxLayout(thumbs_frame)
        thumbs_layout.addWidget(QLabel("Highlights"))
        self.thumb_grid = QGridLayout()
        thumbs_layout.addLayout(self.thumb_grid)
        center_panel.addWidget(thumbs_frame, 1)

        content.addLayout(center_panel, 2)

        # --- Right: Metrics & Coach feed ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        metrics_frame = QFrame()
        metrics_frame.setProperty("class", "card")
        metrics_layout = QVBoxLayout(metrics_frame)

        metrics_layout.addWidget(QLabel("Live Metrics"))

        grid = QGridLayout()
        grid.setSpacing(8)

        self.metric_widgets = {}
        metric_labels = [
            ("Bat Speed", "bat_speed"),
            ("Arm Speed", "arm_speed"),
            ("Elbow (deg)", "elbow"),
            ("Knee (deg)", "knee"),
            ("Weight", "weight"),
            ("Track ID", "track_id"),
        ]

        for idx, (title, key) in enumerate(metric_labels):
            t = QLabel(title)
            t.setStyleSheet("color: #9fb4c8;")
            v = QLabel("--")
            v.setObjectName("metric_" + key)
            v.setProperty("class", "metric")
            v.setStyleSheet("font-size: 20px; font-weight: 700;")
            grid.addWidget(t, idx, 0)
            grid.addWidget(v, idx, 1)
            self.metric_widgets[key] = v

        metrics_layout.addLayout(grid)
        right_panel.addWidget(metrics_frame)

        # coach feed
        coach_frame = QFrame()
        coach_frame.setProperty("class", "card")
        coach_layout = QVBoxLayout(coach_frame)
        coach_layout.addWidget(QLabel("Live Coach (most recent messages)"))
        self.coach_list = QListWidget()
        coach_layout.addWidget(self.coach_list)
        coach_buttons = QHBoxLayout()
        self.btn_clear_coach = QPushButton("Clear")
        self.btn_copy_coach = QPushButton("Copy")
        coach_buttons.addWidget(self.btn_clear_coach)
        coach_buttons.addWidget(self.btn_copy_coach)
        coach_layout.addLayout(coach_buttons)
        right_panel.addWidget(coach_frame, 1)

        # status / logs
        logs_frame = QFrame()
        logs_frame.setProperty("class", "card")
        logs_layout = QVBoxLayout(logs_frame)
        logs_layout.addWidget(QLabel("Activity Log"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(160)
        logs_layout.addWidget(self.log_box)
        right_panel.addWidget(logs_frame)

        content.addLayout(right_panel, 1)

        # Wire buttons
        self.btn_load.clicked.connect(self.load_video)
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_open_highlights.clicked.connect(self.open_highlights)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.chk_coach.clicked.connect(self.toggle_coach)
        self.btn_clear_coach.clicked.connect(self.coach_list.clear)
        self.btn_copy_coach.clicked.connect(self.copy_coach_to_clipboard)

        self.timeline.sliderReleased.connect(self.on_seek)

        # initial populate highlights
        self.refresh_highlights()

    # -------------------------
    # Shortcuts
    # -------------------------
    def install_shortcuts(self):
        # Keyboard shortcuts
        self.btn_load.setShortcut(QKeySequence("L"))
        self.btn_open_highlights.setShortcut(QKeySequence("H"))
        self.btn_pause.setShortcut(QKeySequence(Qt.Key_Space))

    # -------------------------
    # Actions
    # -------------------------
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi *.mov)")
        if not path:
            return
        self.video_path = path
        self.log(f"Loaded: {os.path.basename(path)}")
        self.btn_start.setEnabled(True)
        # try to get total frames for timeline
        try:
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.timeline.setMaximum(max(1, total))
        except Exception:
            pass

    def start_analysis(self):
        if self.video_path is None:
            QMessageBox.warning(self, "No video", "Please load a video first.")
            return

        if CricketEngine is None:
            QMessageBox.critical(self, "Missing Engine", f"Could not import CricketEngine:\n{self.import_error}")
            return

        # disable some controls while running
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_load.setEnabled(False)
        self.fps_spin.setEnabled(False)
        self.target_input.setEnabled(False)

        fps = self.fps_spin.value()
        tid = self.target_input.text().strip() or None

        self.worker = VideoWorker(self.video_path, fps=fps, target_id=tid)
        self.worker.frame_signal.connect(self.update_video)
        self.worker.metric_signal.connect(self.update_metrics)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_worker_error)

        self.worker.start()
        self.status_label.setText("Status: Running")
        self.log("Analysis started")

    def toggle_pause(self):
        if not self.worker:
            return
        if not self.worker.running:
            return
        if self.worker.paused:
            self.worker.resume()
            self.btn_pause.setText("⏸ Pause (Space)")
            self.status_label.setText("Status: Running")
            self.log("Resumed")
        else:
            self.worker.pause()
            self.btn_pause.setText("▶ Resume (Space)")
            self.status_label.setText("Status: Paused")
            self.log("Paused")

    def stop_analysis(self):
        if not self.worker:
            return
        self.worker.stop()
        self.status_label.setText("Status: Stopping...")
        self.log("Stopping analysis...")

    def open_highlights(self):
        folder = self.highlights_dir
        if os.path.exists(folder):
            try:
                if sys.platform.startswith("win"):
                    os.startfile(folder)
                elif sys.platform.startswith("darwin"):
                    os.system(f"open '{folder}'")
                else:
                    os.system(f"xdg-open '{folder}'")
            except Exception:
                QMessageBox.information(self, "Folder", f"Highlights folder: {folder}")
            return
        QMessageBox.warning(self, "Missing", "Highlights folder not found")

    def export_csv(self):
        if not os.path.exists(self.csv_path):
            QMessageBox.warning(self, "CSV Not Found", "No CSV found yet. It will be created during analysis.")
            return
        # ask where to copy
        dest, _ = QFileDialog.getSaveFileName(self, "Export CSV", "cricket_analysis_export.csv", "CSV Files (*.csv)")
        if dest:
            try:
                import shutil
                shutil.copy2(self.csv_path, dest)
                QMessageBox.information(self, "Exported", f"CSV exported to {dest}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not export CSV: {e}")

    def toggle_coach(self):
        if self.chk_coach.isChecked():
            self.chk_coach.setText("Toggle Coach: ON")
            self.log("Coach: ON")
        else:
            self.chk_coach.setText("Toggle Coach: OFF")
            self.log("Coach: OFF")

    def copy_coach_to_clipboard(self):
        text = "\n".join(self.coach_list.item(i).text() for i in range(self.coach_list.count()))
        QApplication.clipboard().setText(text)
        self.log("Coach feed copied to clipboard")

    # -------------------------
    # UI Updaters (called from worker thread via signals)
    # -------------------------
    def update_video(self, qt_img):
        pix = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_metrics(self, data):
        # frame sync
        fid = data.get("frame_id")
        if fid is not None:
            # block signals while setting value to avoid triggering events
            self.timeline.blockSignals(True)
            self.timeline.setValue(int(fid))
            self.timeline.blockSignals(False)

        # update overlay quick metrics
        bat = data.get("bat_speed", "--")
        arm = data.get("arm_speed", "--")
        self.overlay_bat.setText(f"Bat: {bat} km/h" if bat is not None else "Bat: --")
        self.overlay_arm.setText(f"Arm: {arm} km/h" if arm is not None else "Arm: --")
        self.overlay_track.setText(f"TrackID: {data.get('track_id', '--')}")

        # update right-hand metrics cards
        mappings = {
            "bat_speed": "bat_speed",
            "arm_speed": "arm_speed",
            "elbow": "elbow",
            "knee": "knee",
            "weight": "weight",
            "track_id": "track_id"
        }
        for key, widget_key in mappings.items():
            val = data.get(key, "--")
            w = self.metric_widgets.get(widget_key)
            if w:
                w.setText(str(val))

        # coach messaging (if enabled)
        if self.chk_coach.isChecked() and data.get("feedback"):
            msg = data.get("feedback")
            # keep recent small history
            if self.coach_list.count() > 20:
                self.coach_list.takeItem(0)
            self.coach_list.addItem(msg)
            # also append to log
            self.log(f"Coach: {msg}")

        # highlight created this frame?
        highlight = data.get("highlight")
        if highlight:
            self.log(f"Saved highlight: {highlight}")
            # refresh thumbnails to include new highlight
            self.refresh_highlights()

    def on_seek(self):
        val = self.timeline.value()
        if self.worker:
            self.worker.seek(val)
            self.log(f"Seek to frame {val}")

    def on_finished(self):
        self.log("Analysis finished")
        self.status_label.setText("Status: Idle")
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_load.setEnabled(True)
        self.fps_spin.setEnabled(True)
        self.target_input.setEnabled(True)
        self.refresh_highlights()

    def on_worker_error(self, text):
        self.log("Error: " + text)
        QMessageBox.critical(self, "Worker Error", text)

    # -------------------------
    # Highlights / thumbnails
    # -------------------------
    def refresh_highlights(self):
        # clear grid
        while self.thumb_grid.count():
            item = self.thumb_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        files = []
        if os.path.exists(self.highlights_dir):
            for f in sorted(os.listdir(self.highlights_dir), reverse=True):
                if f.lower().endswith((".mp4", ".avi", ".mov")):
                    files.append(os.path.join(self.highlights_dir, f))

        # show up to 6 thumbnails
        max_show = 6
        for idx, path in enumerate(files[:max_show]):
            pix = qpixmap_from_image_path(path, max_size=(240, 140))
            btn = QPushButton()
            btn.setFixedSize(240, 140)
            if pix:
                btn.setIcon(QIcon(pix))
                btn.setIconSize(QSize(240, 140))
            else:
                btn.setText(os.path.basename(path))
            btn.clicked.connect(partial(self.play_highlight, path))
            r = idx // 3
            c = idx % 3
            self.thumb_grid.addWidget(btn, r, c)

    def play_highlight(self, path):
        if not os.path.exists(path):
            QMessageBox.warning(self, "Missing", "Highlight file not found")
            return
        # Play in a lightweight modal-like loop (blocks, but small)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Could not open highlight")
            return

        self.log(f"Playing highlight: {os.path.basename(path)}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            QApplication.processEvents()
            # wait a bit: approximate 30fps playback
            cv2.waitKey(int(1000 / 30))
        cap.release()
        self.log("Highlight playback finished")

    # -------------------------
    # Logging helper
    # -------------------------
    def log(self, text):
        self.log_box.append(text)

    # -------------------------
    # Clean exit
    # -------------------------
    def closeEvent(self, event):
        # stop worker if running
        try:
            if self.worker and self.worker.running:
                self.worker.stop()
        except Exception:
            pass
        event.accept()


# ---------------------------
# Run the App
# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CricketAIApp()
    win.show()
    sys.exit(app.exec_())