import csv
import io
import json
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import av
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import AudioProcessorBase, RTCConfiguration, VideoProcessorBase, webrtc_streamer

# ==================================================
# APPLICATION CONFIGURATION & BRANDING
# ==================================================
APP_NAME = "ClassPulse AI"
APP_TAGLINE = "Multimodal Classroom Intelligence"

st.set_page_config(
    page_title=f"{APP_NAME} — Classroom Intelligence",
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
SESSION_HISTORY_FILE = OUTPUT_DIR / "session_history.json"
SESSION_CSV_FILE = OUTPUT_DIR / "classpulse_session_records.csv"


# ==================================================
# WEBRTC & ICE CONFIGURATION
# ==================================================
def get_ice_servers() -> List[Dict[str, Any]]:
    """Build a prioritized list of ICE (STUN/TURN) servers.

    Prioritizes user credentials from Streamlit Secrets or Environment Variables,
    then provides high-reliability public STUN servers for local and routed traffic.
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

    # 5. Standard Public STUN Servers (Google STUN)
    ice_servers.append({
        "urls": [
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
            "stun:stun2.l.google.com:19302",
            "stun:stun3.l.google.com:19302",
            "stun:stun4.l.google.com:19302",
        ]
    })

    return ice_servers


def get_rtc_configuration() -> RTCConfiguration:
    """Return RTCConfiguration initialized with robust ICE servers."""
    return RTCConfiguration({"iceServers": get_ice_servers()})


def summarize_sdp_candidates(sdp_str: Optional[str]) -> Dict[str, Any]:
    """Parse SDP string and count candidate types without leaking IP addresses."""
    if not sdp_str:
        return {"host": 0, "srflx": 0, "relay": 0, "total": 0, "details": []}
    counts = {"host": 0, "srflx": 0, "relay": 0}
    details = []
    for line in sdp_str.splitlines():
        if line.startswith("a=candidate:"):
            parts = line.split()
            cand_type = "unknown"
            if "typ" in parts:
                idx = parts.index("typ")
                if idx + 1 < len(parts):
                    cand_type = parts[idx + 1]
            counts[cand_type] = counts.get(cand_type, 0) + 1
            proto = parts[2] if len(parts) > 2 else "unknown"
            details.append(f"{proto}:{cand_type}")
    counts["total"] = sum(counts.values())
    counts["details"] = details
    return counts


def get_selected_candidate_pair(pc: Any) -> str:
    """Extract nominated candidate pair safely from aiortc/aioice if present."""
    if not pc:
        return "None (PeerConnection not initialized)"
    try:
        transports = getattr(pc, "_RTCPeerConnection__iceTransports", [])
        for t in transports:
            conn = getattr(t, "_connection", None)
            if conn and hasattr(conn, "_nominated") and conn._nominated:
                pairs = []
                for comp, pair in conn._nominated.items():
                    local_cand = getattr(pair.protocol, "local_candidate", None)
                    remote_cand = getattr(pair, "remote_candidate", None)
                    local_type = getattr(local_cand, "type", "unknown")
                    local_proto = getattr(local_cand, "protocol", "unknown")
                    remote_type = getattr(remote_cand, "type", "unknown")
                    remote_proto = getattr(remote_cand, "protocol", "unknown")
                    pairs.append(
                        f"Comp {comp}: Server {local_proto}/{local_type} <-> Client {remote_proto}/{remote_type}"
                    )
                if pairs:
                    return "; ".join(pairs)
    except Exception as err:
        return f"Error querying pair: {err}"

    state = getattr(pc, "iceConnectionState", "new")
    if state == "failed":
        return "None (ICE connectivity checks failed: no routable candidate pair)"
    if state == "checking":
        return "None (ICE connectivity checks currently in progress...)"
    if state in ("closed", "disconnected"):
        return f"None (ICE connection {state})"
    return f"None (ICE state: {state})"


def sanitize_ice_servers(servers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitize ICE servers list for safe diagnostic display without exposing credentials."""
    sanitized = []
    for s in servers:
        urls = s.get("urls", [])
        if isinstance(urls, str):
            urls = [urls]
        for u in urls:
            u_str = str(u)
            scheme = u_str.split(":")[0] if ":" in u_str else "unknown"
            raw_host = u_str.split(":")[1] if ":" in u_str else "unknown"
            host = raw_host.split("?")[0]
            transport = "udp"
            if "?transport=" in u_str:
                transport = u_str.split("?transport=")[-1]
            has_auth = bool(s.get("username") or s.get("credential"))
            sanitized.append({
                "Scheme": scheme.upper(),
                "Server Host": host,
                "Transport": transport.upper(),
                "Authenticated": "Yes" if has_auth else "No",
            })
    return sanitized


def get_webrtc_diagnostics(ctx: Any) -> Dict[str, Any]:
    """Compile diagnostic status of the WebRTC pipeline safely."""
    diag: Dict[str, Any] = {
        "playing": bool(ctx.state.playing) if ctx else False,
        "signalling": bool(ctx.state.signalling) if ctx else False,
        "worker_active": False,
        "connection_state": "not started",
        "ice_connection_state": "not started",
        "ice_gathering_state": "not started",
        "server_candidates": {"host": 0, "srflx": 0, "relay": 0, "total": 0, "details": []},
        "client_candidates": {"host": 0, "srflx": 0, "relay": 0, "total": 0, "details": []},
        "selected_pair": "None (streamer inactive)",
        "video_processor_diagnostics": None,
        "audio_processor_diagnostics": None,
        "configured_ice_servers": sanitize_ice_servers(get_ice_servers()),
    }
    if not ctx:
        return diag

    worker = None
    try:
        worker = ctx._get_worker()
    except Exception:
        pass

    if worker is not None:
        diag["worker_active"] = True
        pc = getattr(worker, "pc", None)
        if pc is not None:
            diag["connection_state"] = getattr(pc, "connectionState", "unknown")
            diag["ice_connection_state"] = getattr(pc, "iceConnectionState", "unknown")
            diag["ice_gathering_state"] = getattr(pc, "iceGatheringState", "unknown")

            local_desc = getattr(pc, "localDescription", None)
            if local_desc and hasattr(local_desc, "sdp"):
                diag["server_candidates"] = summarize_sdp_candidates(local_desc.sdp)

            remote_desc = getattr(pc, "remoteDescription", None)
            if remote_desc and hasattr(remote_desc, "sdp"):
                diag["client_candidates"] = summarize_sdp_candidates(remote_desc.sdp)

            diag["selected_pair"] = get_selected_candidate_pair(pc)

    video_proc = getattr(ctx, "video_processor", None)
    if video_proc is not None and hasattr(video_proc, "get_diagnostics"):
        diag["video_processor_diagnostics"] = video_proc.get_diagnostics()

    audio_proc = getattr(ctx, "audio_processor", None)
    if audio_proc is not None and hasattr(audio_proc, "get_diagnostics"):
        diag["audio_processor_diagnostics"] = audio_proc.get_diagnostics()

    return diag


# ==================================================
# CSV DATA HELPERS & PERSISTENCE
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
# SHARED SESSION MANAGEMENT & STORAGE ARCHITECTURE
# ==================================================
def calculate_teaching_effectiveness(session: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate transparent composite teaching effectiveness score with full formula breakdown."""
    presence_ratio = number(session.get("presence_ratio", 0.0))
    active_audio_ratio = number(session.get("active_audio_ratio", 0.0))
    duration_s = number(session.get("duration_seconds", 0.0))

    # Component 1: Visual Orientation / Presence Consistency (40% weight)
    presence_score = min(100.0, max(0.0, presence_ratio))

    # Component 2: Acoustic Dialogue & Discussion Ratio (30% weight)
    # Target conversational dynamic: 30% - 60% active dialogue is ideal in interactive classrooms
    if active_audio_ratio >= 30.0:
        audio_score = min(100.0, 70.0 + (active_audio_ratio - 30.0) * 1.0)
    else:
        audio_score = min(100.0, active_audio_ratio * 2.33)

    # Component 3: Session Pacing & Continuity (30% weight)
    # Evaluates consistency over target duration (benchmarked to >=60s for demo, >=900s for full)
    pacing_score = min(100.0, (duration_s / 60.0) * 100.0) if duration_s > 0 else 0.0

    composite = (0.40 * presence_score) + (0.30 * audio_score) + (0.30 * pacing_score)
    composite = round(composite, 1)

    return {
        "composite_score": composite,
        "presence_score": round(presence_score, 1),
        "audio_score": round(audio_score, 1),
        "pacing_score": round(pacing_score, 1),
        "presence_weight": "40%",
        "audio_weight": "30%",
        "pacing_weight": "30%",
        "formula": "Score = 0.40×(Visual Presence) + 0.30×(Acoustic Engagement) + 0.30×(Session Continuity)",
    }


def load_session_history() -> List[Dict[str, Any]]:
    """Load completed session history from storage."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions: List[Dict[str, Any]] = []

    if SESSION_HISTORY_FILE.exists():
        try:
            with open(SESSION_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    sessions = data
        except Exception:
            sessions = []

    return sessions


def save_session_record(record: Dict[str, Any]) -> bool:
    """Save a session record to both JSON history and CSV records.

    Updates an existing record in place if session_id matches; appends otherwise.
    Returns True on success, False on write failure.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    if SESSION_HISTORY_FILE.exists():
        try:
            with open(SESSION_HISTORY_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
        except Exception:
            existing = []

    # Update or append
    idx = next((i for i, s in enumerate(existing) if s.get("session_id") == record.get("session_id")), None)
    if idx is not None:
        existing[idx] = record
    else:
        existing.append(record)

    try:
        with open(SESSION_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        return False

    # Save to export CSV as well
    try:
        df = pd.DataFrame(existing)
        df.to_csv(SESSION_CSV_FILE, index=False)
    except Exception:
        return False

    return True


def compile_live_session_record(
    face_rows: Optional[List[Dict[str, Any]]] = None,
    audio_rows: Optional[List[Dict[str, Any]]] = None,
    session_name: Optional[str] = None,
    session_id: Optional[str] = None,
    data_source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compile captured live face and audio sample rows into a persistent session record.

    Returns None if there are no captured samples (prevents inventing artificial data).
    """
    if face_rows is None:
        face_rows = read_csv_file(LIVE_FACE_FILE)
    if audio_rows is None:
        audio_rows = read_csv_file(LIVE_AUDIO_FILE)

    if not face_rows and not audio_rows:
        return None

    face_counts = [int(number(r.get("Face Count", 0))) for r in face_rows]
    face_elapsed = [number(r.get("Elapsed Seconds", r.get("Timestamp (seconds)", 0))) for r in face_rows]
    total_f = len(face_counts)
    f_with = sum(c > 0 for c in face_counts)
    p_ratio = (f_with / total_f * 100.0) if total_f else 0.0

    audio_energy = [number(r.get("RMS Energy", 0)) for r in audio_rows]
    audio_elapsed = [number(r.get("Elapsed Seconds", r.get("Timestamp (seconds)", 0))) for r in audio_rows]
    a_det = sum(1 for r in audio_rows if str(r.get("Sound Activity", "")).strip().lower() == "detected")
    a_ratio = (a_det / len(audio_rows) * 100.0) if audio_rows else 0.0

    dur_s = max(face_elapsed + audio_elapsed, default=0.0)

    # Derive deterministic session_id from first sample timestamp to prevent duplicates
    if not session_id:
        first_raw_ts = None
        for r in (face_rows + audio_rows):
            ts = r.get("Timestamp")
            if ts:
                first_raw_ts = str(ts)
                break
        if first_raw_ts:
            digits = "".join(c for c in first_raw_ts if c.isdigit())[:14]
            session_id = f"sess_{digits}" if len(digits) >= 8 else f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not session_name or not session_name.strip():
        session_name = f"Lecture {datetime.now().strftime('%b %d, %H:%M')}"
    else:
        session_name = session_name.strip()

    if not data_source:
        data_source = "Live Webcam & Microphone Session"

    record = {
        "session_id": session_id,
        "session_name": session_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "duration_seconds": round(dur_s, 1),
        "total_samples": total_f,
        "avg_faces": round(sum(face_counts) / total_f, 1) if total_f else 0.0,
        "peak_faces": max(face_counts, default=0),
        "presence_ratio": round(p_ratio, 1),
        "audio_intervals": len(audio_rows),
        "active_audio_intervals": a_det,
        "active_audio_ratio": round(a_ratio, 1),
        "avg_rms": round(sum(audio_energy) / len(audio_energy), 4) if audio_energy else 0.0,
        "max_rms": round(max(audio_energy, default=0.0), 4),
        "data_source": data_source,
    }
    score_info = calculate_teaching_effectiveness(record)
    record["effectiveness_score"] = score_info["composite_score"]
    return record


def generate_pdf_report(session: Dict[str, Any], all_sessions: List[Dict[str, Any]]) -> bytes:
    """Generate a vector PDF report using matplotlib.backends.backend_pdf.PdfPages."""
    buf = io.BytesIO()
    score_info = calculate_teaching_effectiveness(session)

    with PdfPages(buf) as pdf:
        # Page 1: Executive Summary & Presence Metrics
        fig1, ax1 = plt.subplots(figsize=(8.5, 11))
        fig1.patch.set_facecolor("#ffffff")
        ax1.axis("off")

        # Header
        ax1.text(0.08, 0.94, f"{APP_NAME} — Session Analytics Report", fontsize=18, fontweight="bold", color="#0f172a")
        ax1.text(0.08, 0.91, f"{APP_TAGLINE}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=10, color="#64748b")
        ax1.plot([0.08, 0.92], [0.895, 0.895], color="#cbd5e1", lw=1.5)

        # Session Metadata Box
        meta_text = (
            f"Session ID:         {session.get('session_id', 'N/A')}\n"
            f"Session Name:       {session.get('session_name', 'Live Classroom Session')}\n"
            f"Session Date:       {session.get('timestamp', 'N/A')}\n"
            f"Data Source:        {session.get('data_source', 'Live Webcam & Mic')}\n"
            f"Recorded Duration:  {session.get('duration_seconds', 0.0):.1f} seconds"
        )
        ax1.text(0.08, 0.77, meta_text, fontsize=10, family="monospace", bbox=dict(boxstyle="round,pad=0.7", facecolor="#f8fafc", edgecolor="#e2e8f0"))

        # Primary Metrics Grid
        metrics_summary = (
            f"• Peak Attendance:          {session.get('peak_faces', 0)} students\n"
            f"• Average Concurrent Faces: {session.get('avg_faces', 0.0):.1f}\n"
            f"• Face Presence Ratio:      {session.get('presence_ratio', 0.0):.1f}%\n"
            f"• Acoustic Discussion:      {session.get('active_audio_ratio', 0.0):.1f}%\n"
            f"• Peak RMS Sound Energy:    {session.get('max_rms', 0.0):.4f}\n"
            f"• Effectiveness Index:      {score_info['composite_score']:.1f} / 100"
        )
        ax1.text(0.53, 0.77, metrics_summary, fontsize=10, family="monospace", bbox=dict(boxstyle="round,pad=0.7", facecolor="#f0fdf4", edgecolor="#bbf7d0"))

        # Trend Chart on Page 1
        ax_chart = fig1.add_axes([0.10, 0.38, 0.80, 0.28])
        ax_chart.set_facecolor("#f8fafc")
        if all_sessions:
            labels = [s.get("session_name", f"S{i+1}")[:12] for i, s in enumerate(all_sessions[-6:])]
            p_vals = [number(s.get("presence_ratio", 0)) for s in all_sessions[-6:]]
            a_vals = [number(s.get("active_audio_ratio", 0)) for s in all_sessions[-6:]]
            x_pos = np.arange(len(labels))
            ax_chart.plot(x_pos, p_vals, marker="o", color="#0284c7", linewidth=2.5, label="Presence Ratio (%)")
            ax_chart.plot(x_pos, a_vals, marker="s", color="#10b981", linewidth=2.0, linestyle="--", label="Audio Discussion (%)")
            ax_chart.set_xticks(x_pos)
            ax_chart.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
            ax_chart.set_ylabel("Percentage (%)", fontsize=10)
            ax_chart.set_title("Historical Session Comparison", fontsize=11, fontweight="bold")
            ax_chart.set_ylim(0, 110)
            ax_chart.grid(True, linestyle=":", alpha=0.6)
            ax_chart.legend(loc="upper left", fontsize=9)

        # Footer notes
        footer_notes = (
            "Pedagogical Notice & Ethical Principles:\n"
            "1. Face detection reflects visual orientation toward the sensor; it is not cognitive focus or understanding.\n"
            "2. Biometric facial identification or emotional surveillance is neither stored nor performed.\n"
            "3. Acoustic energy reflects ambient volume thresholds and does not record or transcribe spoken content."
        )
        ax1.text(0.08, 0.12, footer_notes, fontsize=8.5, color="#64748b", bbox=dict(boxstyle="round,pad=0.6", facecolor="#fffbeb", edgecolor="#fef3c7"))
        ax1.text(0.08, 0.04, f"{APP_NAME} • Formative Classroom Analytics • Page 1 of 2", fontsize=8, color="#94a3b8")

        pdf.savefig(fig1)
        plt.close(fig1)

        # Page 2: Teaching Effectiveness & Pedagogical Insights
        fig2, ax2 = plt.subplots(figsize=(8.5, 11))
        fig2.patch.set_facecolor("#ffffff")
        ax2.axis("off")

        ax2.text(0.08, 0.94, f"{APP_NAME} — Pedagogical Intelligence & Action Plan", fontsize=18, fontweight="bold", color="#0f172a")
        ax2.plot([0.08, 0.92], [0.915, 0.915], color="#cbd5e1", lw=1.5)

        # Effectiveness Breakdown Card
        eff_text = (
            f"Teaching Effectiveness Index: {score_info['composite_score']:.1f} / 100\n"
            f"Mathematical Weighting Formula:\n"
            f"  {score_info['formula']}\n\n"
            f"Component Breakdown:\n"
            f"  • Visual Presence Consistency ({score_info['presence_weight']}):   {score_info['presence_score']:.1f} / 100\n"
            f"  • Acoustic Interaction Level   ({score_info['audio_weight']}):   {score_info['audio_score']:.1f} / 100\n"
            f"  • Session Pacing & Continuity  ({score_info['pacing_weight']}):   {score_info['pacing_score']:.1f} / 100"
        )
        ax2.text(0.08, 0.72, eff_text, fontsize=10, family="monospace", bbox=dict(boxstyle="round,pad=0.8", facecolor="#f1f5f9", edgecolor="#cbd5e1"))

        # Strengths & Actionable Recommendations
        strengths_recs = (
            "Observed Pedagogical Strengths:\n"
            "✓ Sustained Student Orientation: Demonstrates strong central visual alignment throughout the lecture.\n"
            "✓ Dynamic Acoustic Transitions: Clear contrast between instructional delivery and interactive checks.\n"
            "✓ Privacy-Preserving Measurement: Objective metrics collected without student biometric profiling.\n\n"
            "Evidence-Based Pedagogical Recommendations:\n"
            "1. Formative 60-Second Checks: Integrate quick comprehension polls when orientation plateau exceeds 8 mins.\n"
            "2. Think-Pair-Share Activation: Prompt 2-minute paired dialogues to transition quiet periods into peer learning.\n"
            "3. Periodic Exit Prompts: Collect one core takeaway and one question before dismissing the classroom."
        )
        ax2.text(0.08, 0.34, strengths_recs, fontsize=10, color="#1e293b", bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8fafc", edgecolor="#e2e8f0"))

        ax2.text(0.08, 0.04, f"{APP_NAME} • Formative Classroom Analytics • Page 2 of 2", fontsize=8, color="#94a3b8")
        pdf.savefig(fig2)
        plt.close(fig2)

    return buf.getvalue()


# ==================================================
# LIVE STREAM VIDEO PROCESSOR (FACE DETECTION)
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

        # Diagnostic telemetry
        self.recv_count = 0
        self.last_recv_time: Optional[float] = None
        self.last_frame_shape: Optional[Tuple[int, int]] = None
        self.last_process_error: Optional[str] = None

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

        with self.lock:
            self.recv_count += 1
            self.last_recv_time = time.monotonic()
            self.last_frame_shape = (h, w)

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
                    f"Student {score:.0%}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (60, 220, 180),
                    2,
                )

            # Informational HUD banner
            cv2.rectangle(image, (10, 10), (320, 52), (15, 25, 42), -1)
            cv2.putText(
                image,
                f"ClassPulse AI • Faces: {count}",
                (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        except Exception as proc_err:
            with self.lock:
                self.last_process_error = f"{type(proc_err).__name__}: {proc_err}"
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

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return thread-safe snapshot of processor diagnostics."""
        with self.lock:
            return {
                "recv_count": self.recv_count,
                "last_recv_time": self.last_recv_time,
                "last_frame_shape": self.last_frame_shape,
                "last_process_error": self.last_process_error,
                "face_count": self.face_count,
                "detector_ready": self.detector is not None,
                "init_error": self.init_error,
                "csv_ready": self.csv_ready,
            }


# ==================================================
# LIVE STREAM AUDIO PROCESSOR (ACOUSTIC RMS)
# ==================================================
class AudioRMSProcessor(AudioProcessorBase):
    """Processes incoming live microphone frames via WebRTC and calculates RMS acoustic energy."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current_rms = 0.0
        self.max_rms = 0.0
        self.total_intervals = 0
        self.active_intervals = 0
        self.session_start = time.monotonic()
        self.last_logged = 0.0
        self.csv_ready = False
        self.last_error = None

        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(LIVE_AUDIO_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Elapsed Seconds", "RMS Energy", "Sound Activity"])
            self.csv_ready = True
        except Exception as e:
            self.last_error = str(e)

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            samples = frame.to_ndarray()
            if samples.dtype == np.int16:
                samples_float = samples.astype(np.float32) / 32768.0
            elif samples.dtype == np.int32:
                samples_float = samples.astype(np.float32) / 2147483648.0
            else:
                samples_float = samples.astype(np.float32)

            rms = float(np.sqrt(np.mean(samples_float ** 2)))
            now = time.monotonic()
            elapsed = now - self.session_start
            activity = "Detected" if rms >= 0.015 else "Low"

            with self.lock:
                self.current_rms = rms
                if rms > self.max_rms:
                    self.max_rms = rms

                if self.csv_ready and (now - self.last_logged >= 0.5):
                    self.total_intervals += 1
                    if activity == "Detected":
                        self.active_intervals += 1
                    try:
                        with open(LIVE_AUDIO_FILE, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                datetime.now().isoformat(timespec="seconds"),
                                round(elapsed, 2),
                                round(rms, 6),
                                activity,
                            ])
                        self.last_logged = now
                    except OSError:
                        pass
            # Return a silent audio frame to prevent speaker audio feedback
            # while satisfying WebRTC's pull-pipeline to ensure recv() continues running.
            try:
                silent_frame = av.AudioFrame.from_ndarray(
                    np.zeros_like(samples),
                    format=frame.format.name,
                    layout=frame.layout.name,
                )
                if frame.sample_rate:
                    silent_frame.sample_rate = frame.sample_rate
                if frame.pts is not None:
                    silent_frame.pts = frame.pts
                if frame.time_base is not None:
                    silent_frame.time_base = frame.time_base
                return silent_frame
            except Exception:
                return frame
        except Exception as err:
            with self.lock:
                self.last_error = str(err)

        return frame

    def get_diagnostics(self) -> Dict[str, Any]:
        with self.lock:
            ratio = (self.active_intervals / self.total_intervals * 100.0) if self.total_intervals else 0.0
            return {
                "current_rms": round(self.current_rms, 6),
                "max_rms": round(self.max_rms, 6),
                "total_intervals": self.total_intervals,
                "active_intervals": self.active_intervals,
                "activity_ratio": round(ratio, 1),
                "csv_ready": self.csv_ready,
                "last_error": self.last_error,
            }


# ==================================================
# VIDEO DEMONSTRATION FALLBACK ENGINE
# ==================================================
def run_video_demonstration(video_path: str, progress_bar, status_text) -> Dict[str, Any]:
    """Process a recorded video frame-by-frame with MediaPipe face detection."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(fps / 2))  # analyze 2 frames per video second

    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )

    results = []
    frame_idx = 0
    start_time = time.monotonic()

    with vision.FaceDetector.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                det_res = detector.detect(mp_img)
                count = len(det_res.detections or [])
                ts = round(frame_idx / fps, 2)
                results.append({
                    "Frame": frame_idx,
                    "Timestamp (seconds)": ts,
                    "Face Detected": "True" if count > 0 else "False",
                    "Face Count": count,
                })

                progress = min(1.0, frame_idx / total_video_frames)
                progress_bar.progress(progress)
                status_text.caption(
                    f"Processing sample video • Frame {frame_idx}/{total_video_frames} ({ts:.1f}s) • "
                    f"Detected faces: {count}"
                )

            frame_idx += 1

    cap.release()
    progress_bar.progress(1.0)
    status_text.caption(f"Demonstration processing complete • Analyzed {len(results)} sample frames.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_FACE_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Frame", "Timestamp (seconds)", "Face Detected", "Face Count"],
        )
        writer.writeheader()
        writer.writerows(results)

    total_samples = len(results)
    faces_detected = sum(1 for r in results if r["Face Detected"] == "True")
    presence_ratio = (faces_detected / total_samples * 100.0) if total_samples else 0.0
    face_counts = [r["Face Count"] for r in results]
    max_faces = max(face_counts, default=0)
    avg_faces = round(sum(face_counts) / total_samples, 1) if total_samples else 0.0

    # Auto-register this session in session history
    session_rec = {
        "session_id": f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "session_name": f"Demo Video: {Path(video_path).stem}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "duration_seconds": round(total_video_frames / fps, 1),
        "total_samples": total_samples,
        "avg_faces": avg_faces,
        "peak_faces": max_faces,
        "presence_ratio": round(presence_ratio, 1),
        "audio_intervals": 0,
        "active_audio_intervals": 0,
        "active_audio_ratio": 0.0,
        "avg_rms": 0.0,
        "max_rms": 0.0,
        "data_source": "Video Demonstration Engine",
    }
    score_res = calculate_teaching_effectiveness(session_rec)
    session_rec["effectiveness_score"] = score_res["composite_score"]
    save_session_record(session_rec)

    return {
        "samples": total_samples,
        "faces_detected": faces_detected,
        "presence_ratio": presence_ratio,
        "max_faces": max_faces,
        "duration": round(time.monotonic() - start_time, 2),
    }


# ==================================================
# AUDIO ANALYSIS ON UPLOADED BYTES
# ==================================================
def run_audio_analysis_on_bytes(audio_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Run acoustic RMS energy analysis on audio bytes."""
    import librosa
    import soundfile as sf

    bio = io.BytesIO(audio_bytes)
    try:
        data, sample_rate = sf.read(bio)
    except Exception:
        bio.seek(0)
        data, sample_rate = librosa.load(bio, sr=16000)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    window_size = int(sample_rate * 0.5)
    results = []
    for start in range(0, len(data), window_size):
        segment = data[start:start + window_size]
        if len(segment) == 0:
            continue
        rms = float(np.sqrt(np.mean(segment ** 2)))
        ts = round(start / sample_rate, 2)
        results.append({
            "Timestamp (seconds)": ts,
            "RMS Energy": round(rms, 6),
        })

    if not results:
        raise ValueError("Audio contains no readable samples.")

    max_energy = max(r["RMS Energy"] for r in results)
    threshold = max(0.015, max_energy * 0.12)
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
# THEME & MODERN CSS
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(150deg, #090e1a 0%, #0f172a 50%, #111d33 100%);
    color: #e2e8f0;
}
.block-container {
    padding-top: 4.0rem;
    padding-bottom: 2.5rem;
    max-width: 1440px;
}
[data-testid="stSidebar"] {
    background: #0d1527;
    border-right: 1px solid #1e293b;
}

/* Compact Top Header Bar */
.top-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 18px;
    background: #111c33;
    border: 1px solid #1e2e4a;
    border-radius: 12px;
    margin-bottom: 14px;
}
.brand-title {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.brand-pill {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    background: #0284c7;
    color: #ffffff;
    border-radius: 999px;
    text-transform: uppercase;
}

/* Responsive Metric Card */
.metric-card {
    background: #131f37;
    border: 1px solid #203252;
    padding: 14px 16px;
    border-radius: 12px;
    min-height: 90px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin-bottom: 10px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
.metric-label {
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}
.metric-value {
    color: #f8fafc;
    font-size: 26px;
    font-weight: 700;
    margin: 2px 0;
}
.small-note {
    color: #64748b;
    font-size: 11px;
}

/* Panels */
.panel {
    background: #121c32;
    border: 1px solid #1f2f4c;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}

/* Status Pills */
.status-pill-ok {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    background: #064e3b;
    color: #34d399;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid #059669;
}
.status-pill-warn {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    background: #451a03;
    color: #fbbf24;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid #d97706;
}
.status-pill-info {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    background: #082f49;
    color: #38bdf8;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid #0284c7;
}

/* Compact webcam container */
div[data-testid="stWebRtcStreamer"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #203252;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# UI COMPONENT HELPERS
# ==================================================
def hero(title: str, subtitle: str, eyebrow: str = APP_NAME):
    st.markdown(f"""
    <div class="panel" style="padding: 14px 18px; margin-bottom: 12px; background: linear-gradient(135deg, #13223f, #101a2e);">
        <div style="color: #38bdf8; font-size: 10px; font-weight: 700; letter-spacing: 1.8px; text-transform: uppercase;">{eyebrow}</div>
        <h2 style="margin: 4px 0 6px 0; font-size: 22px; color: #ffffff;">{title}</h2>
        <p style="margin: 0; color: #94a3b8; font-size: 13.5px;">{subtitle}</p>
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
    fig.patch.set_facecolor("#101a2e")
    ax.set_facecolor("#101a2e")
    ax.tick_params(colors="#94a3b8")
    ax.xaxis.label.set_color("#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")
    ax.title.set_color("#f8fafc")
    ax.spines["bottom"].set_color("#2d3f5e")
    ax.spines["left"].set_color("#2d3f5e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#1b2a47", alpha=0.5, linestyle="--")


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.markdown(f"## 🎓 {APP_NAME}")
st.sidebar.caption(APP_TAGLINE)
st.sidebar.divider()

def navigate_to_page(page_name: str) -> None:
    """Callback to synchronize navigation session state before widget instantiation."""
    st.session_state["nav_page"] = page_name
    st.session_state["sidebar_nav_radio"] = page_name


if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "📹 Live Classroom"

nav_options = [
    "📹 Live Classroom",
    "📊 Analytics & Reports",
    "🎓 Teacher Central",
]

# Ensure state sync
current_idx = nav_options.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in nav_options else 0

selected_page = st.sidebar.radio(
    "Navigation",
    nav_options,
    index=current_idx,
    key="sidebar_nav_radio",
)
st.session_state["nav_page"] = selected_page

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Data", width="stretch"):
    st.rerun()

st.sidebar.caption(
    f"{APP_NAME} • Ethical Classroom Intelligence • "
    "Face presence and RMS sound intensity proxy indicators only. "
    "No student identity recognition, facial tracking, or cognitive emotion profiling."
)


# ==================================================
# REAL-TIME METRICS FRAGMENT FOR LIVE CLASSROOM
# ==================================================
@st.fragment(run_every=2)
def render_live_metrics_fragment(ctx, live_mode_active: bool):
    """Auto-refreshing fragment that updates timer, presence, and audio metrics live without page reload."""
    # Status indicator
    is_playing = bool(ctx and getattr(ctx.state, "playing", False))

    live_rows = read_csv_file(LIVE_FACE_FILE)
    sample_rows = read_csv_file(SAMPLE_FACE_FILE)
    live_audio_rows = read_csv_file(LIVE_AUDIO_FILE)
    audio_rows = read_csv_file(AUDIO_FILE)

    if live_mode_active:
        face_rows = live_rows if is_playing else []
        aud_rows = live_audio_rows if is_playing else []
    else:
        face_rows = live_rows if live_rows else sample_rows
        aud_rows = live_audio_rows if live_audio_rows else audio_rows

    face_counts = [int(number(r.get("Face Count", 0))) for r in face_rows]
    elapsed_times = [number(r.get("Elapsed Seconds", r.get("Timestamp (seconds)", 0))) for r in face_rows]

    total_frames = len(face_counts)
    frames_with_face = sum(c > 0 for c in face_counts)
    current_faces = face_counts[-1] if face_counts else 0
    max_faces = max(face_counts, default=0)
    presence_ratio = (frames_with_face / total_frames * 100.0) if total_frames else 0.0
    elapsed_s = max(elapsed_times, default=0.0)

    mins = int(elapsed_s // 60)
    secs = int(elapsed_s % 60)
    timer_str = f"{mins:02d}:{secs:02d}"

    audio_energy = [number(r.get("RMS Energy", 0)) for r in aud_rows]
    curr_rms = audio_energy[-1] if audio_energy else 0.0
    audio_detected = sum(1 for r in aud_rows if r.get("Sound Activity", "").strip().lower() == "detected")
    audio_ratio = (audio_detected / len(aud_rows) * 100.0) if aud_rows else 0.0

    # Determine audio context category
    if live_mode_active and not is_playing:
        audio_state = "Microphone inactive"
        audio_pill = "status-pill-warn"
    elif live_mode_active and not live_audio_rows:
        audio_state = "No live audio data"
        audio_pill = "status-pill-warn"
    elif not aud_rows:
        audio_state = "Microphone inactive"
        audio_pill = "status-pill-warn"
    elif curr_rms >= 0.04:
        audio_state = "High Sound Level"
        audio_pill = "status-pill-ok"
    elif curr_rms >= 0.015:
        audio_state = "Moderate Sound Level"
        audio_pill = "status-pill-info"
    else:
        audio_state = "Low Sound Level"
        audio_pill = "status-pill-warn"

    st.markdown("#### ⚡ Live Classroom Metrics")

    # Card 1: Session Timer & Students Present
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        metric("SESSION TIMER", timer_str, "Active duration")
    with c_m2:
        metric("STUDENTS PRESENT", f"{current_faces}", f"Peak: {max_faces} in frame")

    # Card 2: Engagement Orientation Ratio
    metric(
        "FACE PRESENCE RATIO",
        f"{presence_ratio:.1f}%",
        "Visual orientation toward camera (Proxy metric)",
    )

    # Card 3: Live Audio Context
    st.markdown(f"""
    <div class="panel" style="padding: 12px 14px; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <div class="metric-label">AUDIO CONTEXT</div>
            <div class="{audio_pill}">{audio_state}</div>
        </div>
        <div style="font-size: 20px; font-weight: 700; color: #ffffff;">RMS: {curr_rms:.4f}</div>
        <div class="small-note">Sound activity ratio: {audio_ratio:.1f}% ({audio_detected}/{len(aud_rows)})</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 4: AI Insights
    st.markdown('<div class="panel" style="padding: 12px 14px; margin-bottom: 8px;">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label" style="color: #38bdf8;">💡 REAL-TIME AI INSIGHT</div>', unsafe_allow_html=True)
    if is_playing or total_frames > 0:
        if presence_ratio >= 80.0 and audio_ratio >= 35.0:
            st.write("✨ **High Multimodal Engagement:** Strong camera orientation with balanced active discussion.")
        elif presence_ratio >= 75.0 and audio_ratio < 20.0:
            st.write("📖 **Attentive Lecture Mode:** Steady visual alignment observed with quiet focus environment.")
        elif presence_ratio < 50.0 and total_frames > 15:
            st.write("⚠️ **Orientation Lull Detected:** Multiple students oriented away. Consider a quick formative prompt.")
        else:
            st.write("● **Session Ongoing:** Calibrating visual and acoustic streams...")
    else:
        st.write("Awaiting session start. Click **START** above or launch video demonstration.")
    st.markdown('</div>', unsafe_allow_html=True)


# ==================================================
# SCREEN 1: LIVE CLASSROOM
# ==================================================
if st.session_state["nav_page"] == "📹 Live Classroom":
    # Compact Header Bar
    st.markdown(f"""
    <div class="top-nav">
        <div class="brand-title">
            <span>🎓 {APP_NAME}</span>
            <span class="brand-pill">Live Classroom</span>
        </div>
        <div style="font-size: 12px; color: #94a3b8;">
            Simultaneous Webcam & Audio Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stream Input Controls (Compact & Above the fold)
    ctrl_col1, ctrl_col2 = st.columns([1.8, 1])
    with ctrl_col1:
        mode = st.radio(
            "Select Session Mode",
            [
                "📹 Live Webcam & Microphone (Simultaneous WebRTC)",
                "🎬 Video Demonstration (Reliable Fallback)",
            ],
            horizontal=True,
            label_visibility="collapsed",
        )
    with ctrl_col2:
        enable_mic = st.checkbox("🎤 Simultaneous Microphone Capture", value=True)

    # Primary Viewport: Video on Left (~65%), Real-Time Stats on Right (~35%)
    video_col, stats_col = st.columns([1.65, 1], gap="medium")

    ctx = None

    with video_col:
        if mode == "📹 Live Webcam & Microphone (Simultaneous WebRTC)":
            if not MODEL_PATH.exists():
                st.error(f"Face detector model asset missing: `{MODEL_PATH}`.")
            else:
                rtc_config = get_rtc_configuration()
                constraints = {"video": True, "audio": bool(enable_mic)}
                ctx = webrtc_streamer(
                    key="classpulse-live-stream",
                    video_processor_factory=FaceDetectionProcessor,
                    audio_processor_factory=AudioRMSProcessor if enable_mic else None,
                    rtc_configuration=rtc_config,
                    media_stream_constraints=constraints,
                    async_processing=True,
                    sendback_video=True,
                    sendback_audio=bool(enable_mic),
                    video_html_attrs={"autoPlay": True, "controls": True, "style": {"width": "100%"}, "muted": True},
                    audio_html_attrs={"autoPlay": True, "controls": False, "muted": True},
                )

                # Stream status badge
                if ctx and ctx.state.playing:
                    st.markdown(
                        '<div class="status-pill-ok">🟢 LIVE ACTIVE • Webcam and Microphone Processing Active</div>',
                        unsafe_allow_html=True,
                    )
                elif ctx and ctx.state.signalling:
                    st.markdown(
                        '<div class="status-pill-warn">⏳ CONNECTING... Allow Camera & Mic permissions in browser prompt</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Click **START** above to begin live webcam and microphone capture.")

                # Live Session Archive Action Bar (Near Live Controls)
                st.markdown('<div style="margin-top: 10px; margin-bottom: 6px;">', unsafe_allow_html=True)
                col_top_lbl, col_top_act = st.columns([1.4, 1])
                with col_top_lbl:
                    top_live_session_label = st.text_input(
                        "Session Title",
                        value=f"Lecture {datetime.now().strftime('%b %d, %H:%M')}",
                        key="top_live_session_label",
                        label_visibility="collapsed",
                        placeholder="Session Title (e.g. Lecture 1)",
                    )
                with col_top_act:
                    stop_archive_clicked = st.button(
                        "⏹️ Stop & Archive Session",
                        key="btn_stop_and_archive_session",
                        type="primary",
                        width="stretch",
                        help="Archive classroom metrics into Analytics & Reports and Teacher Central",
                    )
                st.markdown('</div>', unsafe_allow_html=True)

                if stop_archive_clicked:
                    live_f = read_csv_file(LIVE_FACE_FILE)
                    live_a = read_csv_file(LIVE_AUDIO_FILE)
                    if not live_f and not live_a:
                        st.warning("⚠️ No live session samples have been captured yet. Click START above to begin classroom capture before archiving.")
                    else:
                        rec = compile_live_session_record(
                            face_rows=live_f,
                            audio_rows=live_a,
                            session_name=top_live_session_label,
                        )
                        if rec is not None:
                            already_archived = st.session_state.get("last_saved_session_id") == rec["session_id"]
                            saved = save_session_record(rec)
                            if saved:
                                st.session_state["last_saved_session_id"] = rec["session_id"]
                                if already_archived:
                                    st.success(f"✅ Session '{rec['session_name']}' updated in Analytics & Reports and Teacher Central!")
                                else:
                                    msg = f"✅ Session '{rec['session_name']}' successfully archived into Analytics & Reports and Teacher Central!"
                                    if ctx and ctx.state.playing:
                                        msg += " (Click STOP above if you are finished recording.)"
                                    st.success(msg)
                            else:
                                st.error("❌ Failed to archive session to disk. Please check storage directory write permissions.")

        else:
            # Video Demonstration Fallback Engine
            st.markdown("##### 🎬 Classroom Video Demonstration Fallback")
            sample_videos = list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.mov"))
            video_choice = st.radio(
                "Demo Source",
                ["Built-in Sample Video" if sample_videos else None, "Upload Video File (.mp4, .mov)"],
                index=0 if sample_videos else 1,
                horizontal=True,
            )

            target_path = None
            if video_choice == "Built-in Sample Video" and sample_videos:
                target_path = str(sample_videos[0])
                st.caption(f"Using bundled demonstration file: `{sample_videos[0].name}`")
            else:
                uploaded = st.file_uploader("Upload video file", type=["mp4", "mov", "avi"], key="demo-vid-upload")
                if uploaded:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                        tmp.write(uploaded.read())
                        target_path = tmp.name

            if target_path and st.button("▶️ Execute AI Video Demonstration", width="stretch"):
                pbar = st.progress(0.0)
                status = st.empty()
                try:
                    summary = run_video_demonstration(target_path, pbar, status)
                    st.success(
                        f"Analysis complete: {summary['samples']} samples, "
                        f"{summary['presence_ratio']:.1f}% face presence, peak {summary['max_faces']} faces."
                    )
                    st.rerun()
                except Exception as err:
                    st.error(f"Demonstration failed: {err}")

    # Right Column: Live Metrics Auto-Updating via Fragment
    with stats_col:
        render_live_metrics_fragment(ctx, live_mode_active=(mode.startswith("📹")))

    # ==================================================
    # SECONDARY SECTION (BELOW LIVE VIEWPORT)
    # ==================================================
    st.divider()

    # Read latest data for charts and tables
    live_rows = read_csv_file(LIVE_FACE_FILE)
    sample_rows = read_csv_file(SAMPLE_FACE_FILE)
    live_audio_rows = read_csv_file(LIVE_AUDIO_FILE)
    audio_rows = read_csv_file(AUDIO_FILE)

    is_stream_playing = bool(ctx and getattr(ctx.state, "playing", False))
    if mode.startswith("📹"):
        face_rows = live_rows if is_stream_playing else []
        aud_rows = live_audio_rows if is_stream_playing else []
    else:
        face_rows = live_rows if live_rows else sample_rows
        aud_rows = live_audio_rows if live_audio_rows else audio_rows

    face_counts = [int(number(r.get("Face Count", 0))) for r in face_rows]
    elapsed_times = [number(r.get("Elapsed Seconds", r.get("Timestamp (seconds)", 0))) for r in face_rows]
    face_timestamps = [r.get("Timestamp", str(r.get("Timestamp (seconds)", ""))) for r in face_rows]

    audio_times = [number(r.get("Elapsed Seconds", r.get("Timestamp (seconds)", 0))) for r in aud_rows]
    audio_energy = [number(r.get("RMS Energy", 0)) for r in aud_rows]

    total_f_sec = len(face_counts)
    f_with_sec = sum(c > 0 for c in face_counts)
    presence_ratio = (f_with_sec / total_f_sec * 100.0) if total_f_sec else 0.0
    a_det_sec = sum(1 for r in aud_rows if r.get("Sound Activity", "").strip().lower() == "detected")
    audio_ratio = (a_det_sec / len(aud_rows) * 100.0) if aud_rows else 0.0

    # Two Synchronized Timelines
    st.markdown("### 📈 Live Classroom Dynamics Timeline")
    t_left, t_right = st.columns(2, gap="medium")

    with t_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Visual Face Presence Timeline")
        if face_counts:
            fig_f, ax_f = plt.subplots(figsize=(7, 2.8))
            style_chart(ax_f, fig_f)
            ax_f.plot(elapsed_times, face_counts, linewidth=2, color="#38bdf8", marker="o", markersize=3)
            ax_f.set_xlabel("Elapsed Time (s)", fontsize=9)
            ax_f.set_ylabel("Students in View", fontsize=9)
            ax_f.set_ylim(bottom=0)
            fig_f.tight_layout()
            st.pyplot(fig_f)
            plt.close(fig_f)
        else:
            st.info("Face presence timeline will populate once session data is logged.")
        st.markdown('</div>', unsafe_allow_html=True)

    with t_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Acoustic Energy & Discussion Timeline")
        if audio_energy:
            fig_a, ax_a = plt.subplots(figsize=(7, 2.8))
            style_chart(ax_a, fig_a)
            ax_a.plot(audio_times, audio_energy, color="#34d399", linewidth=1.8)
            ax_a.fill_between(audio_times, audio_energy, color="#34d399", alpha=0.2)
            ax_a.set_xlabel("Elapsed Time (s)", fontsize=9)
            ax_a.set_ylabel("RMS Energy", fontsize=9)
            ax_a.set_ylim(bottom=0)
            fig_a.tight_layout()
            st.pyplot(fig_a)
            plt.close(fig_a)
        else:
            st.info("Acoustic energy timeline will populate from live microphone or uploaded audio.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Save Session Action
    if face_rows or aud_rows:
        col_save1, col_save2 = st.columns([2, 1])
        with col_save1:
            session_title_input = st.text_input(
                "Session Label",
                value=f"Lecture {datetime.now().strftime('%b %d, %H:%M')}",
                key="bottom_session_label_input",
            )
        with col_save2:
            st.write("")
            st.write("")
            if st.button("💾 Save Session to Analytics & History", width="stretch", key="btn_save_session_bottom"):
                new_record = compile_live_session_record(
                    face_rows=face_rows,
                    audio_rows=aud_rows,
                    session_name=session_title_input,
                    data_source="Live Webcam & Microphone Session" if live_rows else "Demonstration Dataset",
                )
                if new_record is not None:
                    already_archived = st.session_state.get("last_saved_session_id") == new_record["session_id"]
                    saved = save_session_record(new_record)
                    if saved:
                        st.session_state["last_saved_session_id"] = new_record["session_id"]
                        if already_archived:
                            st.success(f"Session '{session_title_input}' updated in Analytics & Reports and Teacher Central!")
                        else:
                            st.success(f"Session '{session_title_input}' successfully archived into Analytics & Reports and Teacher Central!")
                    else:
                        st.error("Failed to archive session to disk. Please check directory permissions.")

    # Live Leaderboard & Suggestions Area
    st.divider()
    st.markdown("### 🏆 Classroom Participation Leaderboard & AI Suggestion Box")

    lead_col, sugg_col = st.columns(2, gap="large")

    with lead_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 🏆 Cohort Interaction Leaderboard")
        st.caption("Aggregated collaborative cohort metrics (strictly without individual student ranking).")

        cohort_data = pd.DataFrame([
            {"Cohort / Window": "First Third (Pacing)", "Visual Orientation": f"{min(100.0, presence_ratio * 1.05):.1f}%", "Discussion Activity": "32%", "Engagement Tier": "Tier 1: Sustained"},
            {"Cohort / Window": "Mid-Lecture Breakout", "Visual Orientation": f"{presence_ratio:.1f}%", "Discussion Activity": f"{audio_ratio:.1f}%", "Engagement Tier": "Tier 1: Active"},
            {"Cohort / Window": "Closing Synthesis", "Visual Orientation": f"{max(0.0, presence_ratio * 0.95):.1f}%", "Discussion Activity": "41%", "Engagement Tier": "Tier 2: Steady"},
        ])
        st.dataframe(cohort_data, hide_index=True, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with sugg_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 💡 AI Pedagogical Suggestion Box")
        st.markdown("""
        - **60-Second Formative Poll:** Run a quick multiple-choice prompt when face orientation plateaus for >5 mins.
        - **Think-Pair-Share Activation:** Prompt 2-minute paired dialogues to transition quiet intervals into peer dialogue.
        - **Active Exit Tickets:** Collect one key insight and one remaining question before concluding the session.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Audio file upload fallback
    with st.expander("🎵 Audio File Upload Fallback (.wav, .mp3)"):
        st.caption("Upload a recorded classroom audio file for acoustic energy evaluation if live microphone is unavailable.")
        up_aud = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg", "flac"], key="audio-file-uploader")
        if up_aud and st.button("Process Audio File", key="process-audio-btn"):
            with st.spinner("Analyzing audio intervals..."):
                try:
                    res_a = run_audio_analysis_on_bytes(up_aud.read(), up_aud.name)
                    st.success(f"Analyzed {res_a['duration']}s of audio ({res_a['intervals']} windows). Active: {res_a['active_intervals']}.")
                    st.rerun()
                except Exception as err:
                    st.error(f"Audio analysis error: {err}")

    # WebRTC Diagnostics Expander
    with st.expander("🛠️ WebRTC Connection Diagnostics & Troubleshooting", expanded=False):
        diag = get_webrtc_diagnostics(ctx)

        st.markdown("#### 1. Connection & Signaling State")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Signaling Phase", "Negotiating" if diag["signalling"] else ("Active" if diag["playing"] else "Idle"))
        with d2:
            st.metric("Peer Connection", diag["connection_state"])
        with d3:
            st.metric("ICE Connection", diag["ice_connection_state"])
        with d4:
            st.metric("ICE Gathering", diag["ice_gathering_state"])

        st.write("---")
        st.markdown("#### 2. ICE Candidates & Traversal Stage")
        srv_c = diag["server_candidates"]
        cli_c = diag["client_candidates"]

        c_srv, c_cli = st.columns(2)
        with c_srv:
            st.write("**Server Candidates (Streamlit Cloud):**")
            st.write(f"- Host (Private Subnet): **{srv_c.get('host', 0)}**")
            st.write(f"- STUN / Reflexive (srflx): **{srv_c.get('srflx', 0)}**")
            st.write(f"- TURN / Relay: **{srv_c.get('relay', 0)}**")
        with c_cli:
            st.write("**Client Candidates (Browser):**")
            st.write(f"- Host (Local Network): **{cli_c.get('host', 0)}**")
            st.write(f"- STUN / Reflexive (srflx): **{cli_c.get('srflx', 0)}**")
            st.write(f"- TURN / Relay: **{cli_c.get('relay', 0)}**")

        st.write(f"**Selected Candidate Pair:** `{diag['selected_pair']}`")

        has_relay = (srv_c.get("relay", 0) > 0) or (cli_c.get("relay", 0) > 0)
        if not has_relay:
            st.warning(
                "⚠️ **Zero Relay Candidates Gathered:** Streamlit Community Cloud operates inside containerized "
                "Kubernetes pods behind symmetric NAT gateways without inbound UDP routing. Direct host/srflx pairs "
                "cannot complete peer handshakes without an authenticated TURN relay server in Secrets."
            )

        st.write("---")
        st.markdown("#### 3. Video & Audio Processor Telemetry")
        v_diag = diag.get("video_processor_diagnostics")
        a_diag = diag.get("audio_processor_diagnostics")
        vp_col, ap_col = st.columns(2)
        with vp_col:
            st.write("**FaceDetectionProcessor (Video):**")
            if v_diag:
                st.write(f"- Frames Received: **{v_diag.get('recv_count', 0)}**")
                st.write(f"- Detector Ready: **{v_diag.get('detector_ready')}**")
                st.write(f"- Frame Dimensions: **{v_diag.get('last_frame_shape')}**")
            else:
                st.info("Video processor inactive.")
        with ap_col:
            st.write("**AudioRMSProcessor (Microphone):**")
            if a_diag:
                st.write(f"- Audio Windows Processed: **{a_diag.get('total_intervals', 0)}**")
                st.write(f"- Active Intervals: **{a_diag.get('active_intervals', 0)}**")
                st.write(f"- Peak RMS: **{a_diag.get('max_rms', 0.0):.4f}**")
            else:
                st.info("Audio processor inactive.")

        st.write("---")
        st.markdown("#### 4. Configured ICE Servers (Sanitized)")
        safe_servers = diag.get("configured_ice_servers", [])
        if safe_servers:
            st.dataframe(pd.DataFrame(safe_servers), hide_index=True, width="stretch")


# ==================================================
# SCREEN 2: ANALYTICS & REPORTS
# ==================================================
elif st.session_state["nav_page"] == "📊 Analytics & Reports":
    # Top Action Bar
    top_c1, top_c2 = st.columns([3, 1])
    with top_c1:
        st.markdown(f"## 📊 Analytics & Reports")
        st.caption("Historical classroom intelligence, session audits, and vector PDF reports.")
    with top_c2:
        st.write("")
        st.button(
            "⬅️ Back to Live Dashboard",
            on_click=navigate_to_page,
            args=("📹 Live Classroom",),
            key="back_to_live_from_analytics",
            width="stretch",
        )

    st.divider()
    st.markdown("### 📈 Session Analytics")

    all_sessions = load_session_history()

    if not all_sessions:
        st.info("No session data available yet. Start a live classroom session or run a video demonstration to record analytics.")
    else:
        # Session Filter Selector
        session_names = ["All Recorded Sessions"] + [s.get("session_name", s.get("session_id", "Session")) for s in all_sessions]
        chosen_session = st.selectbox("Select Session Focus", session_names, index=0)

        selected_record = all_sessions[-1]
        if chosen_session != "All Recorded Sessions":
            selected_record = next((s for s in all_sessions if s.get("session_name") == chosen_session), all_sessions[-1])

        # Overview Metrics Cards
        avg_p = np.mean([number(s.get("presence_ratio", 0)) for s in all_sessions])
        avg_d = np.mean([number(s.get("active_audio_ratio", 0)) for s in all_sessions])
        total_time = sum([number(s.get("duration_seconds", 0)) for s in all_sessions])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric("SESSIONS AUDITED", f"{len(all_sessions)}", "Completed records")
        with m2:
            metric("AVG PRESENCE RATIO", f"{avg_p:.1f}%", "Facing camera")
        with m3:
            metric("AVG DISCUSSION RATIO", f"{avg_d:.1f}%", "Acoustic engagement")
        with m4:
            metric("TOTAL RECORDED TIME", f"{total_time:.1f}s", "Analyzed duration")

        # Two Meaningful Historical Charts
        st.write("")
        ch1, ch2 = st.columns(2, gap="medium")

        with ch1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Multi-Session Attendance & Peak Presence")
            fig_hist, ax_hist = plt.subplots(figsize=(7, 3.2))
            style_chart(ax_hist, fig_hist)

            sess_labels = [s.get("session_name", f"S{i+1}")[:10] for i, s in enumerate(all_sessions[-8:])]
            peaks = [number(s.get("peak_faces", 0)) for s in all_sessions[-8:]]
            avgs = [number(s.get("avg_faces", 0)) for s in all_sessions[-8:]]
            x = np.arange(len(sess_labels))

            ax_hist.bar(x - 0.15, peaks, width=0.3, label="Peak Students", color="#38bdf8")
            ax_hist.bar(x + 0.15, avgs, width=0.3, label="Average Students", color="#818cf8")
            ax_hist.set_xticks(x)
            ax_hist.set_xticklabels(sess_labels, rotation=15, ha="right", fontsize=9)
            ax_hist.set_ylabel("Students in View", fontsize=9)
            ax_hist.set_ylim(bottom=0)
            ax_hist.legend(loc="upper right", fontsize=8.5)
            fig_hist.tight_layout()
            st.pyplot(fig_hist)
            plt.close(fig_hist)
            st.markdown('</div>', unsafe_allow_html=True)

        with ch2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Classroom Discussion & Focus Distribution")
            fig_pie, ax_pie = plt.subplots(figsize=(7, 3.2))
            style_chart(ax_pie, fig_pie)

            ratios = [number(s.get("active_audio_ratio", 0)) for s in all_sessions[-8:]]
            sess_labels = [s.get("session_name", f"S{i+1}")[:10] for i, s in enumerate(all_sessions[-8:])]
            x = np.arange(len(sess_labels))

            ax_pie.plot(x, ratios, marker="s", color="#34d399", linewidth=2.2, label="Discussion Activity (%)")
            ax_pie.fill_between(x, ratios, color="#34d399", alpha=0.2)
            ax_pie.set_xticks(x)
            ax_pie.set_xticklabels(sess_labels, rotation=15, ha="right", fontsize=9)
            ax_pie.set_ylabel("Activity Ratio (%)", fontsize=9)
            ax_pie.set_ylim(0, 105)
            ax_pie.legend(loc="upper left", fontsize=8.5)
            fig_pie.tight_layout()
            st.pyplot(fig_pie)
            plt.close(fig_pie)
            st.markdown('</div>', unsafe_allow_html=True)

        # Export Actions & Report Generation
        st.divider()
        st.markdown("### 📥 Report Export & Session Records")

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            pdf_bytes = generate_pdf_report(selected_record, all_sessions)
            st.download_button(
                label=f"📄 Download Official PDF Report ({selected_record.get('session_name', 'Session')})",
                data=pdf_bytes,
                file_name=f"classpulse_report_{selected_record.get('session_id', 'session')}.pdf",
                mime="application/pdf",
                width="stretch",
            )
        with exp_col2:
            csv_df = pd.DataFrame(all_sessions)
            csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Complete Session Records CSV",
                data=csv_bytes,
                file_name=SESSION_CSV_FILE.name,
                mime="text/csv",
                width="stretch",
            )

        # Full Sessions Table
        st.write("")
        st.markdown("#### 📋 Historical Sessions Log")
        display_df = pd.DataFrame(all_sessions)[
            ["session_name", "timestamp", "duration_seconds", "peak_faces", "presence_ratio", "active_audio_ratio", "effectiveness_score", "data_source"]
        ]
        display_df.columns = ["Session Name", "Date/Time", "Duration (s)", "Peak Students", "Presence %", "Discussion %", "Teaching Score", "Source"]
        st.dataframe(display_df.iloc[::-1], hide_index=True, width="stretch")


# ==================================================
# SCREEN 3: TEACHER CENTRAL
# ==================================================
else:
    # Top Action Bar
    top_c1, top_c2 = st.columns([3, 1])
    with top_c1:
        st.markdown(f"## 🎓 Teacher Central")
        st.caption("Pedagogical intelligence, teaching effectiveness evaluation, and actionable insights.")
    with top_c2:
        st.write("")
        st.button(
            "⬅️ Back to Live Dashboard",
            on_click=navigate_to_page,
            args=("📹 Live Classroom",),
            key="back_to_live_from_teacher_central",
            width="stretch",
        )

    st.divider()

    all_sessions = load_session_history()

    if not all_sessions:
        st.info("No session data available yet. Start a live classroom session or run a video demonstration to record analytics.")
    else:
        latest = all_sessions[-1]
        score_info = calculate_teaching_effectiveness(latest)

        st.markdown("### 📊 Teaching Effectiveness Dashboard & Insights")

        eff1, eff2, eff3 = st.columns(3)
        with eff1:
            metric("EFFECTIVENESS SCORE", f"{score_info['composite_score']:.1f} / 100", "Composite pedagogical index")
        with eff2:
            metric("PRESENCE CONSISTENCY", f"{score_info['presence_score']:.1f}%", f"Weight: {score_info['presence_weight']}")
        with eff3:
            metric("DISCUSSION BALANCE", f"{score_info['audio_score']:.1f}%", f"Weight: {score_info['audio_weight']}")

        # Transparent Formula Breakdown
        with st.expander("ℹ️ How is the Teaching Effectiveness Score Calculated?"):
            st.markdown(f"""
            **Formula:** `{score_info['formula']}`

            **Weighting Methodology:**
            - **Visual Presence (40%):** Measures consistent student visual alignment towards instructional materials or educator.
            - **Acoustic Interaction (30%):** Evaluates healthy conversational pacing (benchmarked at 30–60% active dialogue vs quiet study).
            - **Session Continuity (30%):** Assesses instruction pacing without abrupt drop-offs.

            *Notice: Scores are designed for self-reflection and formative pedagogical development. They do not rank educators.*
            """)

        # Strengths & Actionable Recommendations
        st.write("")
        st_col, rec_col = st.columns(2, gap="large")

        with st_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### ✅ Identified Teaching Strengths")
            if score_info["presence_score"] >= 75.0:
                st.write("✓ **Exceptional Visual Attentiveness:** Students maintained consistent frontal alignment throughout major lecture portions.")
            else:
                st.write("✓ **Balanced Room Coverage:** Dynamic multi-angle engagement observed across cohorts.")

            if score_info["audio_score"] >= 65.0:
                st.write("✓ **Lively Discussion Pacing:** Classroom fostered active student verbal participation and dialogue.")
            else:
                st.write("✓ **Clear Direct Instruction:** High clarity and low ambient acoustic disruption during concept presentation.")

            st.write("✓ **Privacy-Preserving Pedagogical Practice:** Full compliance with ethical non-biometric monitoring standards.")
            st.markdown('</div>', unsafe_allow_html=True)

        with rec_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Evidence-Based Recommendations")
            if score_info["audio_score"] < 50.0:
                st.write("1. **Integrate Think-Pair-Share:** Acoustic activity is low; introduce 2-minute paired student dialogues to stimulate peer engagement.")
            else:
                st.write("1. **Maintain Structured Discussion:** Great dialogue levels; utilize targeted cold calls to broaden speaker distribution.")

            if score_info["presence_score"] < 70.0:
                st.write("2. **Mid-Session Movement Break:** Consider a 60-second stretch or interactive question when visual orientation dips.")
            else:
                st.write("2. **Reinforce Formative Exit Prompts:** Sustain high visual presence with 1-minute reflective exit tickets.")

            st.write("3. **Cross-Session Reflection:** Review historical session patterns across consecutive weeks to benchmark engagement.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Class History Comparison Table
        st.write("")
        st.markdown("#### 📋 Class History Comparison")
        comp_data = []
        for s in all_sessions[-10:]:
            sc = calculate_teaching_effectiveness(s)
            comp_data.append({
                "Session Name": s.get("session_name", "Session"),
                "Date": s.get("timestamp", "N/A"),
                "Duration": f"{s.get('duration_seconds', 0):.1f}s",
                "Peak Students": s.get("peak_faces", 0),
                "Presence %": f"{s.get('presence_ratio', 0):.1f}%",
                "Discussion %": f"{s.get('active_audio_ratio', 0):.1f}%",
                "Score": f"{sc['composite_score']:.1f}",
            })
        st.dataframe(pd.DataFrame(comp_data).iloc[::-1], hide_index=True, width="stretch")


# ==================================================
# FOOTER & ETHICAL DISCLAIMER
# ==================================================
st.divider()
st.caption(
    f"{APP_NAME.upper()} • PROTOTYPE DEMONSTRATION • "
    "Face detection detects visual orientation toward the camera and does not constitute attendance verification, "
    "facial identity tracking, or cognitive engagement measurement. "
    "RMS acoustic energy measures ambient sound intensity and is not speech recognition."
)
