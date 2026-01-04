# 🏏 Cricket AI Analyzer (Cricket AI Studio Pro)

Cricket AI Analyzer is an AI-powered desktop application that performs real-time cricket batting analysis using computer vision and pose estimation. The system extracts biomechanical metrics, estimates bat/arm/ball speed, detects shot phases, generates explainable coaching feedback, and automatically creates highlight clips through a modern PyQt-based user interface.

This project demonstrates the practical application of Artificial Intelligence, Computer Vision, and Sports Biomechanics in performance analysis and coaching assistance.

---

## 🚀 Key Features

- Real-time player detection and pose tracking (YOLOv8 Pose)
- Biomechanical analysis:
  - Elbow angle
  - Knee angle
  - Weight transfer (Front / Back / Balanced)
- Bat, arm, and ball speed estimation
- Explainable AI-based coaching feedback
- Shot phase detection:
  - Backlift
  - Downswing
  - Contact
  - Follow-through
- Automatic highlight generation per detected shot
- CSV-based analytics logging (research & report ready)
- Desktop application built with PyQt5

---
## 🖼️ Application Dashboard

<p align="center">
  <img src="cricket_ai_studio_dashboard.png" alt="Cricket AI Studio Pro Dashboard" width="90%">
</p>

**Cricket AI Studio Pro – Enhanced UI/UX**  
Live pose tracking, real-time metrics, AI coaching feedback, automatic highlights, and analytics logging in a single desktop interface.

## 🧠 System Architecture

The application follows a modular, pipeline-based architecture where each video frame passes through multiple analysis stages.

### High-Level Flow
nput Video
↓
Pose Detection & Tracking (YOLOv8 Pose)
↓
Biomechanical Analysis
↓
Shot Segmentation
↓
Speed Estimation
↓
AI Coaching Feedback
↓
Highlight Generation + CSV Logging
↓
PyQt Desktop UI (Live Metrics & Visualization)


---

## 🧩 Core Modules

- **Pose Engine** (`engine/pose_engine.py`)  
  Handles multi-person pose tracking, target locking, joint angle calculation, and weight transfer estimation.

- **Shot Segmenter** (`engine/shot_segmenter.py`)  
  Detects shot phases, identifies shot start/end, and buffers frames for highlights.

- **Speed Estimator** (`engine/speed_estimator.py`)  
  Estimates bat speed (wrist-based), ball speed (motion tracking), and arm speed (heuristic).

- **Coach Engine** (`engine/coach.py`)  
  Generates human-readable, rule-based coaching feedback using biomechanical thresholds.

- **Highlight Generator** (`engine/highlight_generator.py`)  
  Automatically saves highlight video clips for each completed shot.

- **CSV Logger** (`engine/csv_logger.py`)  
  Logs per-frame analytics including angles, speeds, shot phase, and coaching feedback.

- **Central Engine** (`engine/engine.py`)  
  Orchestrates all modules and acts as the main processing pipeline.

- **UI Layer** (`app.py`)  
  PyQt5-based desktop interface for live visualization, metrics display, coaching feed, highlights, and logs.

---

## 📁 Project Structure
```
CRICKET-ANALYZER/
│
├── engine/
│ ├── init.py
│ ├── engine.py
│ ├── pose_engine.py
│ ├── shot_segmenter.py
│ ├── speed_estimator.py
│ ├── coach.py
│ ├── csv_logger.py
│ └── highlight_generator.py
│
├── highlights/
│ └── Rohit_Sharma/ # Auto-generated shot highlights
│
├── input_videos/ # Sample input videos
│
├── models/
│ └── yolov8n-pose.pt # YOLOv8 pose model
│
├── utils/ # Utility helpers (future use)
│
├── app.py # Main PyQt application
├── requirements.txt
├── cricket_analysis.csv # Auto-generated analytics file
└── README.md

```
---

**⚙️ Installation**

**1️⃣ Clone the Repository**
```bash
git clone https://github.com/<heysudhuu>/cricket-ai-analyzer.git
cd cricket-ai-analyzer
```
**2️⃣ Create a Virtual Environment (Recommended)**

```
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```
**3️⃣ Install Dependencies**
```
pip install -r requirements.txt

```
**📦 Requirements**
```
opencv-python
numpy
PyQt5
ultralytics
lapx
torch
torchvision
```
Ensure that yolov8n-pose.pt is present in the models/ folder or project root.
**
▶️ How to Run**
```
python app.py
```
**Usage**

-Load a cricket video

-(Optional) Enter target player tracking ID

-Start analysis

-View live metrics and coaching feedback

-Review auto-generated highlights

-Export CSV analytics if required

**🎯 Use Cases**

-Cricket performance analysis

-Coaching assistance systems

-Sports biomechanics research

-AI & Computer Vision academic projects

-Portfolio project for internships and higher studies

**👤 Author**

Sudhanshu Yadav
Computer Science Engineering
AI • Computer Vision • Sports Analytics

## 🔗 Connect With Me

<p align="left">
  <a href="https://www.linkedin.com/in/heysudhu" target="_blank">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" width="32" height="32"/>
  </a>
  &nbsp;&nbsp;
  <a href="https://github.com/heysudhuu" target="_blank">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" width="32" height="32"/>
  </a>
  &nbsp;&nbsp;
  <a href="https://www.instagram.com/hey.sudhu/" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" alt="Instagram" width="32" height="32"/>
  </a>
</p>
