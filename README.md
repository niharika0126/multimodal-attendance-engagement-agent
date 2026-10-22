# EduSense AI — Multimodal Attendance & Engagement Agent

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multimodal-attendance-engagement-agent-jcag5dhhf8hjvcgmj2wtkk.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EduSense AI is a multimodal classroom intelligence dashboard designed to monitor visual face presence and acoustic energy in educational environments. Built with **Streamlit**, **MediaPipe**, **OpenCV**, and **streamlit-webrtc**, it provides real-time webcam face detection, recorded session analytics, pedagogical suggestions, and a reliable pre-recorded video demonstration fallback.

---

## 🚀 Key Features

1. **📹 Live Camera AI Face Detection (`streamlit-webrtc`)**:
   - Real-time video processing powered by Google MediaPipe Face Detection (`face_detector.tflite`).
   - Draws real-time bounding boxes with detection confidence and face counts directly on video frames.
   - Cloud-ready ICE/TURN architecture with fallback relays to bypass restrictive symmetric NATs and firewalls on Streamlit Community Cloud.

2. **🎬 Video Demonstration & Fallback Engine**:
   - Guaranteed end-to-end demonstration mode when camera permissions or network firewalls restrict live WebRTC.
   - Allows analyzing pre-recorded classroom video clips (`.mp4`, `.mov`, `.avi`) frame-by-frame with real MediaPipe face detection.
   - Populates session analytics, timelines, and downloadable CSV datasets in real time.

3. **🔊 Acoustic Sound Activity Analysis**:
   - Evaluates RMS sound energy across 0.5-second audio intervals.
   - Detects acoustic activity above a calibrated relative threshold.
   - Supports in-browser audio file upload (`.wav`, `.mp3`) for cloud deployment, plus local real-time microphone recording via `record_live_audio.py`.

4. **🏆 Classroom Insights & Pedagogical Suggestions**:
   - Actionable formative teaching strategies (Think-Pair-Share, Exit Tickets, 60-second comprehension checks).
   - Promotes holistic educator assessment rather than algorithmic surveillance.

5. **📊 Teacher Central**:
   - Session health metrics, data quality indicators, historical logs, and one-click CSV report export.

---

## 🛠️ Architecture & WebRTC Connectivity

### Why WebRTC Requires STUN/TURN on Cloud Platforms

On `localhost`, browser and server share the local network, allowing direct peer-to-peer WebRTC connections. However, in containerized cloud environments like **Streamlit Community Cloud** (AWS containers):
- The container runs behind symmetric NAT with isolated private IP addresses.
- Direct UDP hole punching via standard STUN alone frequently times out because inbound UDP packets cannot be mapped to the dynamic container port.

### How EduSense AI Resolves WebRTC Timeouts:
- **Layered ICE Server Configuration**:
  1. **Google Public STUN** (`stun:stun.l.google.com:19302`) for initial candidate discovery.
  2. **Metered Open Relay STUN & TURN** (`openrelay.metered.ca`) operating on **UDP and TCP ports 80 and 443**. By routing through standard HTTPS port 443, WebRTC can penetrate restrictive firewalls and cloud NATs without requiring paid subscriptions.
  3. **Zero-Code Custom Secrets Support**: Seamlessly supports private TURN credentials (Metered, Twilio, Cloudflare) via Streamlit Secrets or Environment Variables.
- **Robust Video Processor Lifecycle**:
  - MediaPipe LiteRT accelerator initialization handled safely.
  - Pixel bounding boxes strictly clamped to image boundaries with integer conversions.
  - Exception-guarded frame processing ensures transient pipeline hiccups never crash the WebRTC worker.
- **Reliable Fallback**:
  - The **🎬 Video Demonstration** tab provides a guaranteed working demo using real MediaPipe inference even when camera access is blocked.

---

## 📦 Project Structure

```text
classroom-intelligence/
├── dashboard.py                # Main Streamlit application
├── face_detection.py           # Offline video face detection processor
├── live_camera.py              # Standalone OpenCV webcam script
├── analyze_face_presence.py    # CLI face presence reporting tool
├── record_live_audio.py        # Real-time local microphone recorder
├── analyze_audio.py            # CLI acoustic RMS energy analysis tool
├── models/
│   └── face_detector.tflite    # MediaPipe Face Detector model
├── tests/
│   └── test_pipeline.py        # Automated test suite
├── packages.txt                # Linux apt dependencies for Streamlit Cloud
├── requirements.txt            # Python dependencies
├── .gitignore                  # Prevents committing secrets, media, or caches
└── README.md                   # Documentation
```

---

## 💻 Local Setup & Execution

### 1. Clone the repository
```bash
git clone https://github.com/niharika0126/multimodal-attendance-engagement-agent.git
cd multimodal-attendance-engagement-agent
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the automated test suite
```bash
python -m unittest tests/test_pipeline.py
```

### 5. Launch the Streamlit dashboard
```bash
streamlit run dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Streamlit Community Cloud Deployment

1. Push the repository to GitHub:
   ```bash
   git push origin master
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your repository:
   - **Repository:** `niharika0126/multimodal-attendance-engagement-agent`
   - **Branch:** `master`
   - **Main file path:** `dashboard.py`
4. The deployment automatically utilizes:
   - `packages.txt` for Debian system libraries (`libgl1`, `libglib2.0-0`, `ffmpeg`, `libportaudio2`, `portaudio19-dev`).
   - `requirements.txt` for headless-compatible Python wheels.
5. *(Optional)* If your network blocks all public STUN/TURN traffic, you can add private TURN credentials in your Streamlit Cloud **App Settings > Secrets**:
   ```toml
   TURN_URL = "turn:global.relay.metered.ca:80"
   TURN_USERNAME = "your_username"
   TURN_PASSWORD = "your_password"
   ```

---

## ⚖️ Privacy, Ethics & Limitations

- **No Student Identity Profiling**: EduSense AI does not extract facial embeddings, recognize student identities, or store biometric identifiers.
- **Orientation vs. Engagement**: Face presence detects that a student's face is oriented toward the camera. It **cannot** infer cognitive focus, intellectual engagement, comprehension, or emotion.
- **Acoustic Intensity vs. Speech**: Audio analysis computes Root Mean Square (RMS) sound energy thresholds. It does not transcribe speech, evaluate dialogue quality, or identify speakers.
- **Formative Use Only**: This tool is designed to assist educators in reflecting on classroom dynamics, not for punitive attendance enforcement or automated grading.

---

## 📄 License
This project is open-source under the MIT License.
