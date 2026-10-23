"""Automated verification suite for ClassPulse AI.

Tests face detector, WebRTC video/audio processors, ICE server configuration,
audio pipeline, CSV resilience, session persistence, teaching effectiveness,
and PDF report generation.
"""

import csv
import io
import os
import tempfile
import unittest
from pathlib import Path

import av
import cv2
import numpy as np

from dashboard import (
    APP_NAME,
    BASE_DIR,
    MODEL_PATH,
    AudioRMSProcessor,
    FaceDetectionProcessor,
    calculate_teaching_effectiveness,
    generate_pdf_report,
    get_ice_servers,
    get_rtc_configuration,
    get_webrtc_diagnostics,
    load_session_history,
    number,
    read_csv_file,
    run_audio_analysis_on_bytes,
    sanitize_ice_servers,
    save_session_record,
    summarize_sdp_candidates,
)


class TestClassPulsePipeline(unittest.TestCase):
    def test_app_name_branding(self):
        self.assertEqual(APP_NAME, "ClassPulse AI")

    def test_model_file_exists(self):
        self.assertTrue(
            MODEL_PATH.exists(),
            f"Face detector model asset missing at {MODEL_PATH}",
        )
        self.assertGreater(
            MODEL_PATH.stat().st_size,
            10000,
            "Model file size is unexpectedly small.",
        )

    def test_ice_servers_configuration(self):
        # 1. Default configuration includes STUN
        servers = get_ice_servers()
        self.assertIsInstance(servers, list)
        self.assertGreater(len(servers), 0)

        stun_found = any(
            any("stun:" in str(url) for url in s.get("urls", []))
            for s in servers
        )
        self.assertTrue(stun_found, "At least one STUN server must be configured.")

        # 2. When TURN credentials are provided via environment, verify TURN is included
        try:
            os.environ["TURN_URL"] = "turn:relay.example.com:443"
            os.environ["TURN_USERNAME"] = "testuser"
            os.environ["TURN_PASSWORD"] = "testpass"
            servers_with_turn = get_ice_servers()
            turn_found = any(
                any("turn:" in str(url) for url in s.get("urls", []))
                for s in servers_with_turn
            )
            self.assertTrue(turn_found, "TURN server must be configured when credentials are provided.")
        finally:
            os.environ.pop("TURN_URL", None)
            os.environ.pop("TURN_USERNAME", None)
            os.environ.pop("TURN_PASSWORD", None)

        rtc_config = get_rtc_configuration()
        self.assertIsNotNone(rtc_config)

    def test_csv_reader_resilience(self):
        # 1. Non-existent file
        fake_path = BASE_DIR / "output" / "non_existent_file_xyz.csv"
        rows = read_csv_file(fake_path)
        self.assertEqual(rows, [])

        # 2. Number parser
        self.assertEqual(number("12.5"), 12.5)
        self.assertEqual(number(None, default=0.0), 0.0)
        self.assertEqual(number("invalid", default=-1.0), -1.0)

    def test_face_detection_processor_with_frames(self):
        proc = FaceDetectionProcessor()
        self.assertIsNone(proc.init_error)
        self.assertIsNotNone(proc.detector)

        # Create dummy BGR image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(test_img, (320, 240), 60, (200, 200, 200), -1)

        frame = av.VideoFrame.from_ndarray(test_img, format="bgr24")
        out_frame = proc.recv(frame)

        self.assertIsInstance(out_frame, av.VideoFrame)
        out_img = out_frame.to_ndarray(format="bgr24")
        self.assertEqual(out_img.shape, (480, 640, 3))
        self.assertEqual(proc.recv_count, 1)

    def test_audio_rms_processor_with_frames(self):
        proc = AudioRMSProcessor()
        self.assertEqual(proc.total_intervals, 0)

        # Create synthetic audio frame with loud signal (440Hz sine wave)
        frame = av.AudioFrame(format="s16", layout="mono", samples=800)
        t = np.linspace(0, 800 / 16000, 800, endpoint=False)
        data = (0.6 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        for plane in frame.planes:
            plane.update(data.tobytes())

        out_frame = proc.recv(frame)
        self.assertIsInstance(out_frame, av.AudioFrame)
        self.assertGreater(proc.current_rms, 0.01)

        diag = proc.get_diagnostics()
        self.assertIn("current_rms", diag)
        self.assertIn("activity_ratio", diag)

    def test_audio_analysis_with_synthetic_audio(self):
        # Generate 1 second 440Hz sine wave
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        import soundfile as sf
        bio = io.BytesIO()
        sf.write(bio, waveform, sr, format="WAV")
        audio_bytes = bio.getvalue()

        summary = run_audio_analysis_on_bytes(audio_bytes, "synthetic_test.wav")
        self.assertGreater(summary["intervals"], 0)
        self.assertGreater(summary["max_rms"], 0.01)
        self.assertGreater(summary["active_intervals"], 0)

    def test_video_demonstration_engine(self):
        # Create a tiny 1-second video in memory/temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp_path, fourcc, 10.0, (320, 240))
            for _ in range(10):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.circle(frame, (160, 120), 40, (255, 255, 255), -1)
                out.write(frame)
            out.release()

            class DummyProgress:
                def progress(self, val):
                    pass

            class DummyStatus:
                def caption(self, text):
                    pass

            from dashboard import run_video_demonstration
            res = run_video_demonstration(tmp_path, DummyProgress(), DummyStatus())
            self.assertIn("samples", res)
            self.assertGreater(res["samples"], 0)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def test_session_persistence_and_effectiveness(self):
        test_session = {
            "session_id": "test_sess_001",
            "session_name": "Unit Test Lecture",
            "timestamp": "2026-10-23 20:00",
            "duration_seconds": 120.0,
            "total_samples": 60,
            "avg_faces": 3.0,
            "peak_faces": 4,
            "presence_ratio": 85.0,
            "audio_intervals": 20,
            "active_audio_intervals": 10,
            "active_audio_ratio": 50.0,
            "avg_rms": 0.035,
            "max_rms": 0.082,
            "data_source": "Unit Test Pipeline",
        }
        score_info = calculate_teaching_effectiveness(test_session)
        self.assertIn("composite_score", score_info)
        self.assertGreater(score_info["composite_score"], 50.0)
        self.assertIn("formula", score_info)

        test_session["effectiveness_score"] = score_info["composite_score"]
        save_session_record(test_session)

        history = load_session_history()
        self.assertIsInstance(history, list)
        self.assertTrue(any(s.get("session_id") == "test_sess_001" for s in history))

        # Test PDF generation
        pdf_bytes = generate_pdf_report(test_session, history)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)

    def test_webrtc_diagnostics_and_processor_telemetry(self):
        # 1. Test processor telemetry
        proc = FaceDetectionProcessor()
        test_img = np.zeros((240, 320, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(test_img, format="bgr24")
        proc.recv(frame)

        p_diag = proc.get_diagnostics()
        self.assertEqual(p_diag["recv_count"], 1)
        self.assertIsNotNone(p_diag["last_recv_time"])
        self.assertEqual(p_diag["last_frame_shape"], (240, 320))
        self.assertTrue(p_diag["detector_ready"])

        # 2. Test SDP candidate parser
        sample_sdp = (
            "v=0\r\n"
            "a=candidate:1 1 udp 2130706431 10.0.0.1 50000 typ host\r\n"
            "a=candidate:2 1 udp 1694498815 1.2.3.4 50001 typ srflx\r\n"
            "a=candidate:3 1 udp 16777215 5.6.7.8 50002 typ relay\r\n"
        )
        sdp_summary = summarize_sdp_candidates(sample_sdp)
        self.assertEqual(sdp_summary["host"], 1)
        self.assertEqual(sdp_summary["srflx"], 1)
        self.assertEqual(sdp_summary["relay"], 1)

        # 3. Test ICE server sanitization (never leak credentials)
        raw_servers = [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:example.com:443?transport=tcp"],
                "username": "secret_user",
                "credential": "super_secret_password",
            },
        ]
        sanitized = sanitize_ice_servers(raw_servers)
        self.assertEqual(len(sanitized), 2)
        for s in sanitized:
            self.assertNotIn("secret_user", str(s))
            self.assertNotIn("super_secret_password", str(s))
            self.assertIn("Authenticated", s)

        # 4. Test safe diagnostics compilation with None context
        ctx_diag = get_webrtc_diagnostics(None)
        self.assertFalse(ctx_diag["playing"])
        self.assertFalse(ctx_diag["signalling"])
        self.assertEqual(ctx_diag["connection_state"], "not started")


# Backward compatibility alias
TestEduSensePipeline = TestClassPulsePipeline

if __name__ == "__main__":
    unittest.main()
