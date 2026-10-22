import csv
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import av
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

# ==================================================
# CONFIGURATION & PATHS
# ==================================================
st.set_page_config(
    page_title="EduSense AI — Classroom Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "face_detector.tflite"
LIVE_FACE_FILE = OUTPUT_DIR / "live_face_presence.csv"
SAMPLE_FACE_FILE = OUTPUT_DIR / "face_presence.csv"
AUDIO_FILE = OUTPUT_DIR / "audio_activity.csv"
LIVE_AUDIO_FILE = OUTPUT_DIR / "live_audio_activity.csv"


# ==================================================
# WEBRTC & ICE CONFIGURATION
# ==================================================
def get_ice_servers() -> List[Dict[str, Any]]:
    """Build a robust list of ICE (STUN/TURN) servers.

    Prioritizes user credentials from Streamlit Secrets or Environment Variables,
    then provides high-reliability public STUN servers and free Metered Open Relay
    TURN servers (operating on UDP and TCP ports 80/443 to bypass restrictive NATs).
    """
    ice_servers: List[Dict[str, Any]] = []

    # 1. Custom list directly defined in secrets
    try:
        if "ice_servers" in st.secrets:
            custom = st.secrets["ice_servers"]
            if isinstance(custom, list):
                return list(custom)
    except Exception:
        pass

    # 2. Twilio ICE server helper (if configured in secrets)
    try:
        if "TWILIO_ACCOUNT_SID" in st.secrets and "TWILIO_AUTH_TOKEN" in st.secrets:
            from streamlit_webrtc import get_twilio_ice_servers
            return get_twilio_ice_servers(
                st.secrets["TWILIO_ACCOUNT_SID"],
                st.secrets["TWILIO_AUTH_TOKEN"],
            )
    except Exception:
        pass

    # 3. Cloudflare TURN helper (if configured in secrets)
    try:
        if (
            "CLOUDFLARE_TURN_KEY_ID" in st.secrets
            and "CLOUDFLARE_TURN_API_TOKEN" in st.secrets
        ):
            from streamlit_webrtc import get_cloudflare_ice_servers
            return get_cloudflare_ice_servers(
                st.secrets["CLOUDFLARE_TURN_KEY_ID"],
                st.secrets["CLOUDFLARE_TURN_API_TOKEN"],
            )
    except Exception:
        pass

    # 4. Custom TURN URL / Credentials from Secrets or Environment Variables
    turn_url = None
    turn_username = None
    turn_password = None

    try:
        if "TURN_URL" in st.secrets:
            turn_url = st.secrets["TURN_URL"]
            turn_username = st.secrets.get("TURN_USERNAME")
            turn_password = st.secrets.get("TURN_PASSWORD")
    except Exception:
        pass

    if not turn_url:
        turn_url = os.environ.get("TURN_URL")
        turn_username = os.environ.get("TURN_USERNAME")
        turn_password = os.environ.get("TURN_PASSWORD")

    if turn_url:
        turn_entry: Dict[str, Any] = {
            "urls": [turn_url] if isinstance(turn_url, str) else list(turn_url),
        }
        if turn_username:
            turn_entry["username"] = str(turn_username)
        if turn_password:
            turn_entry["credential"] = str(turn_password)
        ice_servers.append(turn_entry)

    # 5. Standard Public STUN Servers
    ice_servers.extend([
        {
            "urls": [
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302",
                "stun:stun2.l.google.com:19302",
                "stun:stun3.l.google.com:19302",
                "stun:stun4.l.google.com:19302",
            ]
        },
        {
            "urls": [
                "stun:openrelay.metered.ca:80",
            ]
        },
    ])

    # 6. Metered Open Relay Free TURN Servers (UDP + TCP on standard ports 80/443)
    # Bypasses symmetric NATs and firewalls on cloud container environments
    ice_servers.append({
        "urls": [
            "turn:openrelay.metered.ca:80",
            "turn:openrelay.metered.ca:443",
            "turn:openrelay.metered.ca:443?transport=tcp",
            "turn:openrelay.metered.ca:80?transport=tcp",
        ],
        "username": "openrelayproject",
        "credential": "openrelayproject",
    })

    return ice_servers


def get_rtc_configuration() -> RTCConfiguration:
    """Return RTCConfiguration initialized with robust ICE servers."""
    return RTCConfiguration({"iceServers": get_ice_servers()})


# ==================================================
# CSV DATA HELPERS
# ==================================================
def read_csv_file(path: Path) -> List[Dict[str, str]]:
    """Read a CSV safely; return an empty list if missing or unreadable."""
    if not path.exists():
        return []

    try:
        with open(path, newline="", encoding="utf-8-sig") as file:
            return list(csv.DictReader(file))
    except (OSError, csv.Error):
        return []


def number(value: Any, default: float = 0.0) -> float:
    """Convert value to float safely."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ==================================================
# LIVE FACE DETECTOR PROCESSOR
# ==================================================
class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.face_count = 0
        self.detector = None
        self.session_start = time.monotonic()
        self.last_recorded = 0.0
        self.csv_ready = False
        self.init_error = None

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

            options = vision.FaceDetectorOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(MODEL_PATH)
                ),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=0.5,
            )
            self.detector = vision.FaceDetector.create_from_options(options)

            # Reset live session CSV for fresh camera run
            with open(LIVE_FACE_FILE, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Elapsed Seconds", "Face Count"])
            self.csv_ready = True
        except Exception as err:
            self.init_error = str(err)
            self.detector = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        h, w = image.shape[:2]

        if self.detector is None:
            err_msg = self.init_error or "Detector unavailable"
            cv2.rectangle(image, (10, 10), (min(w - 10, 480), 60), (0, 0, 180), -1)
            cv2.putText(
                image,
                f"Error: {err_msg[:45]}",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)
            detections = result.detections or []
            count = len(detections)
            now = time.monotonic()
            elapsed = now - self.session_start

            with self.lock:
                self.face_count = count
                # Save at most 1 measurement per second to keep log lightweight
                if self.csv_ready and (now - self.last_recorded >= 1.0):
                    try:
                        with open(
                            LIVE_FACE_FILE, "a", newline="", encoding="utf-8"
                        ) as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                datetime.now().isoformat(timespec="seconds"),
                                round(elapsed, 2),
                                count,
                            ])
                        self.last_recorded = now
                    except OSError:
                        pass

            for detection in detections:
                box = detection.bounding_box
                x1 = max(0, int(box.origin_x))
                y1 = max(0, int(box.origin_y))
                bw = int(box.width)
                bh = int(box.height)
                x2 = min(w, x1 + bw)
                y2 = min(h, y1 + bh)

                cv2.rectangle(image, (x1, y1), (x2, y2), (60, 220, 180), 2)
                score = (
                    detection.categories[0].score if detection.categories else 0.0
                )
                cv2.putText(
                    image,
                    f"Face {score:.0%}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (60, 220, 180),
                    2,
                )

            # Informational HUD banner
            cv2.rectangle(image, (10, 10), (320, 58), (15, 25, 42), -1)
            cv2.putText(
                image,
                f"Faces detected: {count}",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        except Exception as proc_err:
            cv2.rectangle(image, (10, 10), (360, 50), (0, 140, 255), -1)
            cv2.putText(
                image,
                f"Notice: {type(proc_err).__name__}",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def get_face_count(self) -> int:
        with self.lock:
            return self.face_count


# ==================================================
# VIDEO DEMONSTRATION FALLBACK ENGINE
# ==================================================
def run_video_demonstration(video_path: str, progress_bar, status_text) -> Dict[str, Any]:
    """Process a recorded video frame-by-frame with MediaPipe face detection.

    Saves measurements into output/face_presence.csv and returns summary stats.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Face detector model not found at: {MODEL_PATH}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )

    records = []
    frame_idx = 0
    # Sample every N frames (approx 2 samples per second) for fast responsiveness
    sample_step = max(1, int(fps / 2))

    try:
        with vision.FaceDetector.create_from_options(options) as detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if frame_idx % sample_step != 0 and frame_idx != 1:
                    continue

                timestamp = round((frame_idx - 1) / fps, 2)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = detector.detect(mp_img)
                count = len(results.detections or [])

                records.append({
                    "Frame": frame_idx,
                    "Timestamp (seconds)": timestamp,
                    "Face Detected": "True" if count > 0 else "False",
                    "Face Count": count,
                })

                if total_frames > 0:
                    progress = min(1.0, frame_idx / total_frames)
                    progress_bar.progress(progress)
                    status_text.caption(
                        f"Analyzing video: frame {frame_idx}/{total_frames} ({timestamp:.1f}s) • {count} face(s)"
                    )
    finally:
        cap.release()

    # Save to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_FACE_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Frame", "Timestamp (seconds)", "Face Detected", "Face Count"],
        )
        writer.writeheader()
        writer.writerows(records)

    return {
        "samples": len(records),
        "total_frames": frame_idx,
        "max_faces": max([r["Face Count"] for r in records], default=0),
        "duration": records[-1]["Timestamp (seconds)"] if records else 0.0,
    }


# ==================================================
# AUDIO ANALYSIS ENGINE
# ==================================================
def run_audio_analysis_on_bytes(audio_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Process uploaded audio file and save output/audio_activity.csv."""
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        import soundfile as sf
        data, sample_rate = sf.read(tmp_path)
    except Exception:
        import librosa
        data, sample_rate = librosa.load(tmp_path, sr=None)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Convert to mono float32
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)

    window_size = int(sample_rate * 0.5)
    results = []
    for start in range(0, len(data), window_size):
        seg = data[start:start + window_size]
        if len(seg) == 0:
            continue
        rms = float(np.sqrt(np.mean(seg ** 2)))
        timestamp = round(start / sample_rate, 2)
        results.append({
            "Timestamp (seconds)": timestamp,
            "RMS Energy": round(rms, 6),
        })

    max_energy = max((r["RMS Energy"] for r in results), default=0.0)
    threshold = max(0.01, max_energy * 0.10)

    for r in results:
        r["Sound Activity"] = "Detected" if r["RMS Energy"] >= threshold else "Low"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIO_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Timestamp (seconds)", "RMS Energy", "Sound Activity"]
        )
        writer.writeheader()
        writer.writerows(results)

    active_count = sum(1 for r in results if r["Sound Activity"] == "Detected")
    return {
        "intervals": len(results),
        "duration": round(len(data) / sample_rate, 2),
        "active_intervals": active_count,
        "max_rms": max_energy,
    }


# ==================================================
# THEME & CSS
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #0b1120, #111c32);
    color: #e5edf8;
}
.block-container {
    padding-top: 1.5rem;
    max-width: 1450px;
}
[data-testid="stSidebar"] {
    background: #101a2c;
}
.hero {
    padding: 24px 28px;
    border: 1px solid #293b57;
    border-radius: 18px;
    background: linear-gradient(120deg, #172943, #152137);
    margin-bottom: 22px;
}
.hero h1 {
    margin: 6px 0 8px 0;
    font-size: 28px;
}
.hero p {
    margin: 0;
    color: #b5c7de;
    font-size: 15px;
}
.eyebrow {
    color: #65d6c4;
    font-size: 11px;
    letter-spacing: 2.2px;
    font-weight: 700;
    text-transform: uppercase;
}
.metric-card {
    background: #17243a;
    border: 1px solid #2a3b55;
    padding: 16px 20px;
    border-radius: 14px;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-label {
    color: #9cb1cc;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.metric-value {
    color: #ffffff;
    font-size: 28px;
    font-weight: 700;
    margin: 4px 0;
}
.small-note {
    color: #7d96b5;
    font-size: 12px;
}
.panel {
    background: #131f33;
    border: 1px solid #293b55;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 14px;
}
.status-pill-ok {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    background: #0f3f33;
    color: #65d6c4;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #1a6f5a;
}
.status-pill-warn {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    background: #3f2a0f;
    color: #f6ad55;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid #6f4b1a;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# UI COMPONENTS
# ==================================================
def hero(title: str, subtitle: str, eyebrow: str = "EDUSENSE AI"):
    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric(label: str, value: Any, note: str = ""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="small-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def style_chart(ax, fig):
    fig.patch.set_facecolor("#131f33")
    ax.set_facecolor("#131f33")
    ax.tick_params(colors="#c5d2e3")
    ax.xaxis.label.set_color("#c5d2e3")
    ax.yaxis.label.set_color("#c5d2e3")
    ax.title.set_color("#ffffff")
    ax.spines["bottom"].set_color("#40516a")
    ax.spines["left"].set_color("#40516a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#22334e", alpha=0.5, linestyle="--")


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.markdown("## 🎓 EduSense AI")
st.sidebar.caption("MULTIMODAL CLASSROOM INTELLIGENCE")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "📹 Live Classroom",
        "🏆 Leaderboard & Suggestions",
        "📊 Teacher Central",
    ],
)

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh dashboard", width="stretch"):
    st.rerun()

st.sidebar.caption(
    "Prototype demonstration • Face presence & acoustic activity only • "
    "No student identity recognition or cognitive engagement claims."
)


# ==================================================
# PAGE 1: LIVE CLASSROOM
# ==================================================
if page == "📹 Live Classroom":
    hero(
        "EduSense AI — Live Classroom",
        "Real-time webcam AI face detection with recorded session analytics and video fallback.",
        "● MULTIMODAL INTELLIGENCE",
    )

    # Input Mode Selection: Real Webcam vs Video Demonstration
    mode = st.radio(
        "Select Video Input Mode",
        [
            "📹 Live Webcam Stream (WebRTC)",
            "🎬 Video Demonstration (Reliable Fallback)",
        ],
        horizontal=True,
    )

    if mode == "📹 Live Webcam Stream (WebRTC)":
        st.markdown("### 📹 Live Webcam — AI Face Detection")

        if not MODEL_PATH.exists():
            st.error(
                f"Face detector model asset missing: `{MODEL_PATH}`. "
                "Ensure `models/face_detector.tflite` is present in the repository."
            )
            ctx = None
        else:
            rtc_config = get_rtc_configuration()
            ctx = webrtc_streamer(
                key="live-face-detection",
                video_processor_factory=FaceDetectionProcessor,
                rtc_configuration=rtc_config,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            # Live Stream Connection States
            if ctx.state.playing:
                face_count = 0
                if ctx.video_processor:
                    face_count = ctx.video_processor.get_face_count()
                st.markdown(
                    f'<div class="status-pill-ok">🟢 LIVE STREAMING • {face_count} face(s) currently in view</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Camera stream active. Processed frames display face detection boxes "
                    "and counts. Measurements are logged automatically to CSV."
                )
            elif ctx.state.signalling:
                st.markdown(
                    '<div class="status-pill-warn">⏳ NEGOTIATING WEBRTC CONNECTION...</div>',
                    unsafe_allow_html=True,
                )
                st.info(
                    "Connecting to webcam... ICE servers are establishing peer relay. "
                    "If prompted by your browser, make sure to click **Allow Camera Access**."
                )
            else:
                st.caption(
                    "Click **START** above and allow camera permissions in your browser. "
                    "If your browser or network blocks direct WebRTC peer connections, switch to the "
                    "**Video Demonstration (Reliable Fallback)** tab above."
                )

        # Connection Diagnostics & Troubleshooting Expander
        with st.expander("🛠️ WebRTC Connection Diagnostics & Troubleshooting"):
            st.markdown("""
            **Deployment & Connectivity Status:**
            - **Signaling:** WebSocket over HTTPS (Streamlit Cloud).
            - **ICE Discovery:** Google STUN (`stun.l.google.com:19302`) + Metered Open Relay STUN/TURN active.
            - **TURN Relay:** Enabled for UDP and TCP on ports 80 and 443 to bypass symmetric NATs and restrictive firewalls.
            
            **Troubleshooting WebRTC Timeouts:**
            1. **Camera Permissions:** Ensure your browser has permitted camera access on `*.streamlit.app`.
            2. **Network/Hotspot Restrictions:** Strict enterprise, university, or cellular firewalls may restrict WebRTC ports. 
               - If this happens, use the **🎬 Video Demonstration** tab to test the exact same AI detection pipeline.
            3. **Custom TURN Secrets:** If you have private TURN credentials (Metered, Twilio, Cloudflare), you can store `TURN_URL`, `TURN_USERNAME`, and `TURN_PASSWORD` in your Streamlit Cloud **Secrets** settings without changing code.
            """)

    else:
        # Fallback Demonstration Mode
        st.markdown("### 🎬 Video File Demonstration (Reliable Fallback)")
        st.info(
            "**Demonstration Mode Notice:** Processes a classroom video clip frame-by-frame through the "
            "MediaPipe face detection model. This guarantees a complete end-to-end demo even when "
            "browser permissions or cloud network firewalls restrict live camera streaming."
        )

        sample_videos = list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.mov"))
        video_choice = st.radio(
            "Video Source",
            ["Built-in Sample Video" if sample_videos else None, "Upload a Video File (.mp4, .mov, .avi)"],
            index=0 if sample_videos else 1,
            horizontal=True,
        )

        video_path_to_process = None

        if video_choice == "Built-in Sample Video" and sample_videos:
            selected_sample = st.selectbox(
                "Choose sample video",
                sample_videos,
                format_func=lambda p: p.name,
            )
            video_path_to_process = str(selected_sample)
            st.caption(f"Using server video: `{selected_sample.name}`")
        else:
            uploaded_video = st.file_uploader(
                "Upload a short classroom video clip",
                type=["mp4", "mov", "avi", "webm"],
                help="Upload a video to analyze face presence across all frames.",
            )
            if uploaded_video:
                temp_video = OUTPUT_DIR / f"uploaded_{uploaded_video.name}"
                temp_video.write_bytes(uploaded_video.read())
                video_path_to_process = str(temp_video)
                st.caption(f"Uploaded: `{uploaded_video.name}` ({round(uploaded_video.size / 1024 / 1024, 2)} MB)")

        if video_path_to_process:
            if st.button("▶ Run AI Face Detection on Video", type="primary"):
                prog_bar = st.progress(0.0)
                status_txt = st.empty()
                with st.spinner("Processing video frames through MediaPipe face detector..."):
                    try:
                        res = run_video_demonstration(video_path_to_process, prog_bar, status_txt)
                        st.success(
                            f"✅ Video analysis complete! Analyzed {res['samples']} samples across "
                            f"{res['duration']}s of video. Maximum faces detected: {res['max_faces']}. "
                            "Measurements saved to `output/face_presence.csv`."
                        )
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as err:
                        st.error(f"Video analysis encountered an error: {err}")

    # ==================================================
    # READ ANALYSIS DATA FOR DASHBOARD
    # ==================================================
    live_rows = read_csv_file(LIVE_FACE_FILE)
    sample_rows = read_csv_file(SAMPLE_FACE_FILE)
    audio_rows = read_csv_file(AUDIO_FILE)
    live_audio_rows = read_csv_file(LIVE_AUDIO_FILE)

    has_live_measurements = bool(live_rows)
    face_rows = live_rows if has_live_measurements else sample_rows

    face_counts, elapsed_times, face_timestamps = [], [], []
    for row in face_rows:
        if has_live_measurements:
            elapsed = number(row.get("Elapsed Seconds"))
            timestamp = row.get("Timestamp", "")
        else:
            elapsed = number(row.get("Timestamp (seconds)"))
            timestamp = str(row.get("Timestamp (seconds)", ""))

        face_counts.append(int(number(row.get("Face Count"))))
        elapsed_times.append(elapsed)
        face_timestamps.append(timestamp)

    total_frames = len(face_counts)
    frames_with_face = sum(c > 0 for c in face_counts)
    frames_without_face = total_frames - frames_with_face
    max_faces = max(face_counts, default=0)
    face_percentage = (frames_with_face / total_frames * 100) if total_frames else 0.0
    session_duration = max(elapsed_times, default=0.0)

    # Audio data
    audio_times = [number(r.get("Timestamp (seconds)")) for r in audio_rows]
    audio_energy = [number(r.get("RMS Energy")) for r in audio_rows]
    audio_duration = max(audio_times, default=0.0)

    live_audio_times = [number(r.get("Elapsed Seconds")) for r in live_audio_rows]
    live_audio_energy = [number(r.get("RMS Energy")) for r in live_audio_rows]
    live_audio_duration = max(live_audio_times, default=0.0)
    live_audio_detected = sum(
        1 for r in live_audio_rows
        if r.get("Sound Activity", "").strip().lower() == "detected"
    )

    st.divider()
    st.markdown("### 📊 Session Analytics & Face Presence Metrics")

    if has_live_measurements:
        st.markdown(
            '<div class="status-pill-ok">● DATA SOURCE: Current Live Webcam Session (`live_face_presence.csv`)</div>',
            unsafe_allow_html=True,
        )
    elif sample_rows:
        st.markdown(
            '<div class="status-pill-warn">● DATA SOURCE: Demonstration Video Dataset (`face_presence.csv`)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "No recorded face measurements yet. Start the live webcam or run the video demonstration above."
        )

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("SAMPLES RECORDED", f"{total_frames:,}", "Total analyzed records")
    with c2:
        metric("SAMPLES WITH A FACE", f"{frames_with_face:,}", "At least 1 face detected")
    with c3:
        metric("FACE-PRESENT RATIO", f"{face_percentage:.1f}%", "Orientation presence only")
    with c4:
        metric("SESSION DURATION", f"{session_duration:.1f}s", "Recorded time interval")

    st.write("")
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Face count over time")
        if total_frames > 0:
            fig, ax = plt.subplots(figsize=(8, 3.2))
            style_chart(ax, fig)
            ax.plot(elapsed_times, face_counts, linewidth=2, color="#65d6c4", marker="o", markersize=3)
            ax.set_xlabel("Elapsed time (seconds)")
            ax.set_ylabel("Detected faces")
            ax.set_ylim(bottom=0)
            ax.set_title("AI Face Presence Timeline")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("The timeline chart will render automatically once the first measurement is logged.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Detection Summary")
        st.metric("Peak face count in single frame", max_faces)
        st.metric("Samples with zero faces detected", frames_without_face)
        st.caption(
            "Note: Face detection reflects visual orientation toward the camera. "
            "It does not establish cognitive focus, comprehension, or identity."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ==================================================
    # AUDIO ACTIVITY SECTION
    # ==================================================
    st.divider()
    st.markdown("### 🔊 Audio Activity Analysis")
    st.caption(
        "Acoustic analysis measures RMS energy thresholds to detect sound activity. "
        "It is not speech recognition and does not assess lecture quality."
    )

    a1, a2, a3 = st.columns(3)
    with a1:
        metric("AUDIO SAMPLES", len(audio_energy), "RMS intervals evaluated")
    with a2:
        metric("MAX RMS ENERGY", f"{max(audio_energy, default=0.0):.4f}", "Acoustic intensity")
    with a3:
        metric("AUDIO DURATION", f"{audio_duration:.1f}s", "Audio analysis range")

    if audio_energy:
        fig, ax = plt.subplots(figsize=(10, 2.8))
        style_chart(ax, fig)
        ax.plot(audio_times, audio_energy, color="#8aa7ff", linewidth=1.8)
        ax.fill_between(audio_times, audio_energy, color="#8aa7ff", alpha=0.2)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("RMS sound energy")
        ax.set_title("Recorded Acoustic Energy Over Time")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No recorded audio analysis dataset found in `output/audio_activity.csv`.")

    with st.expander("🎵 Upload & Analyze an Audio Recording (.wav, .mp3)"):
        uploaded_audio = st.file_uploader(
            "Upload audio recording for acoustic analysis",
            type=["wav", "mp3", "ogg", "flac"],
            key="audio-uploader",
        )
        if uploaded_audio:
            if st.button("Analyze Audio File", key="run-audio-analysis"):
                with st.spinner("Calculating RMS energy across 0.5s windows..."):
                    try:
                        res = run_audio_analysis_on_bytes(
                            uploaded_audio.read(), uploaded_audio.name
                        )
                        st.success(
                            f"Analyzed {res['duration']}s of audio ({res['intervals']} intervals). "
                            f"Sound detected in {res['active_intervals']} intervals. Saved to `output/audio_activity.csv`."
                        )
                        st.rerun()
                    except Exception as err:
                        st.error(f"Audio analysis error: {err}")

    # Live Microphone Analysis (from local record_live_audio.py if available)
    if live_audio_energy:
        st.markdown("#### Live Microphone Feed (Local Device Recording)")
        c1, c2, c3 = st.columns(3)
        with c1:
            metric("LIVE SAMPLES", len(live_audio_energy), "0.5s windows")
        with c2:
            metric("ACTIVE INTERVALS", live_audio_detected, "Above RMS threshold")
        with c3:
            metric("RECORDED TIME", f"{live_audio_duration:.1f}s", "Session length")

        fig, ax = plt.subplots(figsize=(10, 2.8))
        style_chart(ax, fig)
        ax.plot(live_audio_times, live_audio_energy, color="#65d6c4", linewidth=1.8)
        ax.fill_between(live_audio_times, live_audio_energy, color="#65d6c4", alpha=0.2)
        ax.set_xlabel("Elapsed time (seconds)")
        ax.set_ylabel("RMS Energy")
        ax.set_title("Live Microphone Acoustic Energy")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Data Table View & Export
    with st.expander("📋 View & Export Raw Face Detection Records"):
        if total_frames:
            df = pd.DataFrame({
                "Timestamp": face_timestamps,
                "Elapsed Seconds": elapsed_times,
                "Face Count": face_counts,
            })
            st.dataframe(df, hide_index=True, width="stretch")
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Face Presence CSV",
                data=csv_data,
                file_name="classroom_face_presence.csv",
                mime="text/csv",
            )
        else:
            st.write("No recorded face data available yet.")


# ==================================================
# PAGE 2: LEADERBOARD & SUGGESTIONS
# ==================================================
elif page == "🏆 Leaderboard & Suggestions":
    hero(
        "Classroom Insights & Suggestions",
        "Formative feedback and practical teaching strategies based on technical observations.",
    )

    st.warning(
        "**Ethical & Scientific Notice:** Individual student tracking and automated engagement scores "
        "are intentionally not implemented. Face presence alone cannot validate comprehension or focus. "
        "This system does not rank students or produce punitive leaderboards."
    )

    live_rows = read_csv_file(LIVE_FACE_FILE)
    sample_rows = read_csv_file(SAMPLE_FACE_FILE)
    face_rows = live_rows if live_rows else sample_rows

    face_counts = [int(number(r.get("Face Count"))) for r in face_rows]
    total_frames = len(face_counts)
    frames_with_face = sum(c > 0 for c in face_counts)
    max_faces = max(face_counts, default=0)
    face_percentage = (frames_with_face / total_frames * 100) if total_frames else 0.0

    st.markdown("### 📊 Current Session Overview")
    if total_frames == 0:
        st.info(
            "No session records found. Visit the **Live Classroom** page to start a webcam "
            "stream or process a demonstration video to generate observations."
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric("OBSERVATIONS LOGGED", f"{total_frames:,}", "Interval records")
    with c2:
        metric("FACE-PRESENT RATIO", f"{face_percentage:.1f}%", "Technical orientation only")
    with c3:
        metric("PEAK FACE COUNT", max_faces, "Maximum concurrent faces")

    st.write("")
    st.markdown("### 💡 Recommended Pedagogical Strategies")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 1. Interactive Understanding Checks")
        st.write(
            "Use quick 60-second formative polls or multiple-choice checks to gather direct, "
            "unambiguous evidence of whether students grasp key concepts."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 2. Think-Pair-Share")
        st.write(
            "Prompt learners to discuss a challenging prompt with a neighbor for 2 minutes before "
            "sharing reflections with the whole room to reactivate participation."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 3. Exit Tickets")
        st.write(
            "Ask students to summarize one core idea they understood and one question that remains "
            "unclear before concluding the session."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 4. Multimodal Synthesis")
        st.write(
            "Combine computer-vision presence signals with direct student feedback and assignments. "
            "Never infer learning efficacy solely from facial orientation or camera gaze."
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ==================================================
# PAGE 3: TEACHER CENTRAL
# ==================================================
else:
    hero(
        "Teacher Central",
        "Session audits, data quality indicators, and exportable analytics for educators.",
    )

    live_rows = read_csv_file(LIVE_FACE_FILE)
    sample_rows = read_csv_file(SAMPLE_FACE_FILE)
    audio_rows = read_csv_file(AUDIO_FILE)
    face_rows = live_rows if live_rows else sample_rows

    face_counts = [int(number(r.get("Face Count"))) for r in face_rows]
    elapsed_times = [
        number(r.get("Elapsed Seconds")) if live_rows else number(r.get("Timestamp (seconds)"))
        for r in face_rows
    ]
    face_timestamps = [
        r.get("Timestamp", "") if live_rows else str(r.get("Timestamp (seconds)", ""))
        for r in face_rows
    ]

    total_frames = len(face_counts)
    frames_with_face = sum(c > 0 for c in face_counts)
    max_faces = max(face_counts, default=0)
    face_percentage = (frames_with_face / total_frames * 100) if total_frames else 0.0
    session_duration = max(elapsed_times, default=0.0)
    audio_energy = [number(r.get("RMS Energy")) for r in audio_rows]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric("SESSION SAMPLES", f"{total_frames:,}", "Validated data points")
    with c2:
        metric("PRESENCE RATIO", f"{face_percentage:.1f}%", "Facing camera")
    with c3:
        metric("RECORDED DURATION", f"{session_duration:.1f}s", "Session length")

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### ✅ Teacher Checklist & Best Practices")
        st.markdown("""
        - **Lighting & Camera Angle:** Confirm the webcam is elevated and positioned with frontal lighting.
        - **Acoustic Calibration:** Position microphones close to discussion areas for clean RMS sampling.
        - **Holistic Assessment:** Blend automated signals with active student feedback and classroom dialogue.
        - **Privacy First:** Reassure students that biometric identity recognition is neither stored nor performed.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 🔍 Data Health & Quality")
        if live_rows:
            st.write("**Active Dataset:** Live Webcam Session (`live_face_presence.csv`)")
        elif sample_rows:
            st.write("**Active Dataset:** Pre-recorded Video Dataset (`face_presence.csv`)")
        else:
            st.write("**Active Dataset:** None (Awaiting session start)")

        st.write(f"- Vision records logged: **{total_frames}**")
        st.write(f"- Maximum simultaneous faces: **{max_faces}**")
        st.write(f"- Audio intervals logged: **{len(audio_energy)}**")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📋 Recent Measurements")
    if total_frames:
        recent_data = pd.DataFrame({
            "Timestamp": face_timestamps,
            "Elapsed Seconds": elapsed_times,
            "Face Count": face_counts,
        })
        st.dataframe(recent_data.tail(25).iloc[::-1], hide_index=True, width="stretch")

        csv_bytes = recent_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Complete Session CSV",
            data=csv_bytes,
            file_name="edusense_session_records.csv",
            mime="text/csv",
        )
    else:
        st.info("Measurements will populate here once a live or demonstration session is executed.")


# ==================================================
# FOOTER & ETHICAL DISCLAIMER
# ==================================================
st.divider()
st.caption(
    "EDUSENSE AI • PROTOTYPE DEMONSTRATION • "
    "Face detection detects visual orientation toward the camera and does not constitute attendance verification, "
    "facial identity tracking, or cognitive engagement measurement. "
    "RMS acoustic energy measures ambient sound intensity and is not speech recognition."
)
