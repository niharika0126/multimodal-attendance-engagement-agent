
import csv
import threading
from pathlib import Path

import av
import cv2
import mediapipe as mp
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==================================================
# CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="EduSense AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = BASE_DIR / "models" / "face_detector.tflite"

LIVE_FACE_FILE = OUTPUT_DIR / "live_face_presence.csv"
SAMPLE_FACE_FILE = OUTPUT_DIR / "face_presence.csv"
AUDIO_FILE = OUTPUT_DIR / "audio_activity.csv"


# ==================================================
# LIVE FACE DETECTOR
# ==================================================
class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.face_count = 0
        self.detector = None

        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )

        self.detector = vision.FaceDetector.create_from_options(options)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = self.detector.detect(mp_image)
        detections = result.detections or []

        with self.lock:
            self.face_count = len(detections)
            count = self.face_count

        for detection in detections:
            box = detection.bounding_box

            x = box.origin_x
            y = box.origin_y
            w = box.width
            h = box.height

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (60, 220, 180),
                2,
            )

            score = detection.categories[0].score

            cv2.putText(
                image,
                f"Face {score:.0%}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (60, 220, 180),
                2,
            )

        cv2.rectangle(image, (10, 10), (300, 58), (15, 25, 42), -1)

        cv2.putText(
            image,
            f"Faces detected: {count}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def get_face_count(self):
        with self.lock:
            return self.face_count


# ==================================================
# THEME
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
    padding: 25px;
    border: 1px solid #293b57;
    border-radius: 18px;
    background: linear-gradient(120deg, #172943, #152137);
    margin-bottom: 20px;
}
.eyebrow {
    color: #65d6c4;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 700;
}
.metric-card {
    background: #17243a;
    border: 1px solid #2a3b55;
    padding: 18px;
    border-radius: 14px;
    min-height: 115px;
}
.metric-label {
    color: #a7bad2;
    font-size: 12px;
}
.metric-value {
    color: #f5f8ff;
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}
.small-note {
    color: #a7bad2;
    font-size: 12px;
}
.panel {
    background: #131f33;
    border: 1px solid #293b55;
    border-radius: 14px;
    padding: 18px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# HELPERS
# ==================================================
def read_csv_file(path):
    if not path.exists():
        return []

    try:
        with open(path, newline="", encoding="utf-8-sig") as file:
            return list(csv.DictReader(file))
    except (OSError, csv.Error):
        return []


def number(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def hero(title, subtitle, eyebrow="EDUSENSE AI"):
    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric(label, value, note=""):
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
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines["bottom"].set_color("#40516a")
    ax.spines["left"].set_color("#40516a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ==================================================
# LOAD RECORDED DATA
# ==================================================
live_rows = read_csv_file(LIVE_FACE_FILE)
sample_rows = read_csv_file(SAMPLE_FACE_FILE)
audio_rows = read_csv_file(AUDIO_FILE)

is_live_data = bool(live_rows)
face_rows = live_rows if is_live_data else sample_rows

face_counts = []
elapsed_times = []
face_timestamps = []

for row in face_rows:
    if is_live_data:
        elapsed = number(row.get("Elapsed Seconds"))
        timestamp = row.get("Timestamp", "")
    else:
        elapsed = number(row.get("Timestamp (seconds)"))
        timestamp = row.get("Timestamp (seconds)", "")

    face_counts.append(int(number(row.get("Face Count"))))
    elapsed_times.append(elapsed)
    face_timestamps.append(timestamp)

total_frames = len(face_counts)
frames_with_face = sum(count > 0 for count in face_counts)
frames_without_face = total_frames - frames_with_face
max_faces = max(face_counts, default=0)

face_percentage = (
    frames_with_face / total_frames * 100
    if total_frames else 0
)

session_duration = max(elapsed_times, default=0)

audio_times = []
audio_energy = []

for row in audio_rows:
    audio_times.append(number(row.get("Timestamp (seconds)")))
    audio_energy.append(number(row.get("RMS Energy")))

audio_duration = max(audio_times, default=0)


# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.markdown("## 🎓 EduSense AI")
st.sidebar.caption("CLASSROOM INTELLIGENCE")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "📹 Live Classroom",
        "🏆 Leaderboard & Suggestions",
        "📊 Teacher Central",
    ],
)

st.sidebar.divider()

if is_live_data:
    st.sidebar.success("Recorded webcam data found")
elif sample_rows:
    st.sidebar.info("Sample-video data found")
else:
    st.sidebar.warning("No recorded face data found")

if st.sidebar.button("🔄 Refresh dashboard", use_container_width=True):
    st.rerun()

st.sidebar.caption(
    "Local prototype • No student identity recognition"
)


# ==================================================
# PAGE 1: LIVE CLASSROOM
# ==================================================
if page == "📹 Live Classroom":

    hero(
        "EduSense AI — Live Classroom",
        "Live webcam face detection with recorded session analytics.",
        "● CLASSROOM INTELLIGENCE",
    )

    st.markdown("### 📹 Live webcam — AI face detection")

    if not MODEL_PATH.exists():
        st.error(
            f"Face detector model not found: {MODEL_PATH}"
        )

    else:
        st.info(
            "Click START and allow camera access in your browser. "
            "The Python backend will detect faces in the live video."
        )

        ctx = webrtc_streamer(
            key="live-face-detection",
            video_processor_factory=FaceDetectionProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            async_processing=True,
        )

        if ctx.state.playing:
            st.success(
                "Camera stream connected. Face boxes and counts "
                "are drawn on each processed frame."
            )
        else:
            st.caption("Camera is stopped. Click START to begin.")

    st.divider()
    st.markdown("### 📊 Recorded session analytics")

    if is_live_data:
        st.success(
            "Historical results are loaded from "
            "output/live_face_presence.csv."
        )
    elif sample_rows:
        st.info(
            "Showing sample-video results. Start and stop a webcam "
            "session with live_camera.py to generate historical webcam data."
        )
    else:
        st.warning(
            "No historical face data yet. Live webcam detection above "
            "can still work independently."
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric("FRAMES RECORDED", f"{total_frames:,}", "Historical data")

    with c2:
        metric("FRAMES WITH A FACE", f"{frames_with_face:,}", "Historical data")

    with c3:
        metric("FACE-PRESENT RATIO", f"{face_percentage:.1f}%", "Not engagement")

    with c4:
        metric("SESSION DURATION", f"{session_duration:.1f}s", "Historical data")

    left, right = st.columns([1.3, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Face count over time")

        if total_frames:
            fig, ax = plt.subplots(figsize=(8, 3.2))
            style_chart(ax, fig)
            ax.plot(
                elapsed_times,
                face_counts,
                linewidth=1.8,
                color="#65d6c4",
            )
            ax.set_xlabel("Elapsed time (seconds)")
            ax.set_ylabel("Detected faces")
            ax.set_ylim(bottom=0)
            ax.set_title("Recorded face detections")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Historical chart appears when recorded data is available.")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Detection summary")
        st.metric("Maximum faces in one frame", max_faces)
        st.metric("Frames without a detected face", frames_without_face)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🔊 Audio analysis")

    a, b, c = st.columns(3)

    with a:
        metric("AUDIO INTERVALS", len(audio_energy), "From audio CSV")

    with b:
        metric(
            "MAX RMS ENERGY",
            f"{max(audio_energy, default=0):.4f}",
            "Sound level measurement",
        )

    with c:
        metric("AUDIO DURATION", f"{audio_duration:.1f}s", "From audio CSV")

    if audio_energy:
        fig, ax = plt.subplots(figsize=(10, 3))
        style_chart(ax, fig)
        ax.plot(audio_times, audio_energy, color="#8aa7ff")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("RMS sound energy")
        ax.set_title("Recorded audio energy")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No audio-analysis data is available yet.")

    st.caption(
        "Face detection does not identify students or prove attendance, "
        "attention, emotion, or engagement. RMS energy is not speech recognition."
    )

    with st.expander("View historical face-count data"):
        if total_frames:
            st.dataframe(
                pd.DataFrame({
                    "Timestamp": face_timestamps,
                    "Elapsed seconds": elapsed_times,
                    "Face count": face_counts,
                }),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.write("No recorded face data available.")


# ==================================================
# PAGE 2: LEADERBOARD & SUGGESTIONS
# ==================================================
elif page == "🏆 Leaderboard & Suggestions":

    hero(
        "Classroom Insights & Suggestions",
        "Review technical signals and practical teaching strategies.",
    )

    st.warning(
        "Individual student tracking and validated engagement scores "
        "are not implemented. This page does not generate a real leaderboard."
    )

    st.markdown("### 📊 Current recorded session")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric("FRAMES ANALYZED", f"{total_frames:,}", "Recorded data")

    with c2:
        metric("FACE-PRESENT RATIO", f"{face_percentage:.1f}%", "Not engagement")

    with c3:
        metric("MAXIMUM FACE COUNT", max_faces, "One frame")

    st.markdown("### 💡 Teaching strategies")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Interactive understanding checks")
        st.write(
            "Use a short poll or quiz to collect direct evidence "
            "of student understanding."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Think-pair-share")
        st.write(
            "Ask students to discuss a problem briefly and explain "
            "their reasoning."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Exit ticket")
        st.write(
            "Ask students to submit one thing they understood "
            "and one question they still have."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Compare sessions carefully")
        st.write(
            "Combine technical measurements with direct feedback. "
            "Do not infer engagement from face presence alone."
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ==================================================
# PAGE 3: TEACHER CENTRAL
# ==================================================
else:

    hero(
        "Teacher Central",
        "Review recorded measurements and practical next steps.",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric("RECORDED FRAMES", f"{total_frames:,}", "Historical data")

    with c2:
        metric("FACE-PRESENT RATIO", f"{face_percentage:.1f}%", "Technical statistic")

    with c3:
        metric("SESSION DURATION", f"{session_duration:.1f}s", "Historical data")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### ✅ Recommended next steps")
        st.markdown("""
        - Check camera placement and lighting.
        - Compare results across recorded sessions.
        - Collect student feedback directly.
        - Keep technical measurements separate from learning assessments.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 🔍 Data quality")

        if is_live_data:
            st.write("**Selected source:** Recorded webcam CSV")
        elif sample_rows:
            st.write("**Selected source:** Sample-video CSV")
        else:
            st.write("**Selected source:** No recorded face data")

        st.write(f"Recorded frames: {total_frames}")
        st.write(f"Maximum faces in one frame: {max_faces}")
        st.write(f"Audio intervals: {len(audio_energy)}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📋 Recent recorded measurements")

    if total_frames:
        recent_data = pd.DataFrame({
            "Timestamp": face_timestamps,
            "Elapsed seconds": elapsed_times,
            "Face count": face_counts,
        })

        st.dataframe(
            recent_data.tail(20).iloc[::-1],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Recorded measurements will appear here after a session.")


# ==================================================
# FOOTER
# ==================================================
st.divider()

st.caption(
    "EDUSENSE AI • LOCAL PROTOTYPE • "
    "Face detection is not identity verification or proof of attendance. "
    "Face presence does not establish attention or engagement."
)
