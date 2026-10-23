# ClassPulse AI — Multimodal Classroom Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multimodal-attendance-engagement-agent-jcag5dhhf8hjvcgmj2wtkk.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ClassPulse AI** is a multimodal classroom intelligence application designed to assist educators in understanding classroom attendance, face orientation trends, and acoustic activity in educational environments. Built with **Streamlit**, **MediaPipe**, **OpenCV**, and **streamlit-webrtc**, it provides simultaneous live webcam and microphone capture, real-time live metric fragments, historical session analytics, vector PDF and CSV reporting, and a reliable pre-recorded video demonstration fallback.

---

## 🚀 Key Features

1. **📹 Live Classroom with Simultaneous Webcam & Microphone Capture**:
   - Single-click session startup activating both webcam and microphone.
   - Real-time video processing powered by Google MediaPipe Face Detection (`face_detector.tflite`) with bounding box HUD.
   - Real-time acoustic energy evaluation computing RMS sound levels from incoming audio frames without playback echo.
   - Auto-updating live metric fragments (`@st.fragment`) keeping session timers, student counts, and audio gauges fresh without manual page reloads.

2. **🎬 Video Demonstration & Fallback Engine**:
   - Guaranteed end-to-end demonstration mode when browser permissions or cloud network firewalls restrict live WebRTC.
   - Analyzes pre-recorded classroom video clips (`.mp4`, `.mov`, `.avi`) frame-by-frame with real MediaPipe face detection.
   - Populates session analytics, timelines, and downloadable CSV datasets in real time.

3. **📊 Analytics & Reports**:
   - Dedicated session analytics screen comparing completed sessions.
   - Multi-session attendance trends and acoustic engagement ratio charts.
   - One-click **PDF Report Generation** using vector graphics and **CSV Data Export**.

4. **🎓 Teacher Central**:
   - Pedagogical intelligence screen featuring an Instructor Profile.
   - Transparent Teaching Effectiveness Dashboard with explained mathematical weighting.
   - Evidence-based pedagogical strengths and actionable teaching recommendations.
   - Historical session audit and comparison table.

5. **🏆 Live Leaderboard & Pedagogical AI Suggestions**:
   - Classroom Participation Leaderboard focusing on collaborative discussion intervals (strictly without punitive individual student profiling).
   - Formative suggestions (Think-Pair-Share, Exit Tickets, 60-second comprehension checks).

---

## 🛠️ Architecture & WebRTC Connectivity

### Why WebRTC Requires STUN/TURN on Cloud Platforms

On `localhost`, browser and server share the local network, allowing direct peer-to-peer WebRTC connections. However, in containerized cloud environments like **Streamlit Community Cloud** (AWS Kubernetes containers):
- The container runs behind a symmetric NAT with isolated private IP addresses.
- Direct UDP hole punching via standard STUN alone frequently times out because inbound unsolicited UDP packets cannot be mapped to dynamic container ports.
- A functional TURN relay server is required for cloud NAT traversal.

### How ClassPulse AI Handles WebRTC & Audio:
- **Layered ICE Server Configuration**:
  1. **Google Public STUN** (`stun:stun.l.google.com:19302`) for initial candidate discovery.
  2. **Configurable TURN Relay via Streamlit Secrets**: Seamlessly supports private or managed TURN credentials (Cloudflare Realtime TURN, Twilio, Hugging Face `HF_TOKEN`, or custom TURN servers).
  3. **Safe Real-Time Diagnostics Panel**: Built-in diagnostics showing signaling state, peer connection state, ICE gathering state, server vs client candidate distribution, and frame counters without leaking tokens or passwords.
- **Echo-Free Audio Pipeline**:
  - `sendback_audio=False` ensures captured microphone audio is analyzed for acoustic intensity without playing back into the educator's speakers.

---

## 📦 Project Structure

```text
classroom-intelligence/
├── dashboard.py                # Main ClassPulse AI Streamlit application (3 Integrated Screens)
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

### 5. Launch the ClassPulse AI dashboard
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
5. *(Optional)* Add private TURN credentials in your Streamlit Cloud **App Settings > Secrets**:
   ```toml
   TURN_URL = "turn:your-turn-server.com:443?transport=tcp"
   TURN_USERNAME = "your_username"
   TURN_PASSWORD = "your_password"
   ```

---

## ⚖️ Privacy, Ethics & Limitations

- **No Student Identity Profiling**: ClassPulse AI does not extract facial embeddings, recognize student identities, or store biometric identifiers.
- **Orientation vs. Engagement**: Face presence detects that a student's face is oriented toward the camera. It **cannot** infer cognitive focus, intellectual engagement, comprehension, or emotion.
- **Acoustic Intensity vs. Speech**: Audio analysis computes Root Mean Square (RMS) sound energy thresholds. It does not transcribe speech, evaluate dialogue quality, or identify speakers.
- **Formative Use Only**: This tool is designed to assist educators in reflecting on classroom dynamics, not for punitive attendance enforcement or automated grading.

---

## 📄 License
This project is open-source under the MIT License.
