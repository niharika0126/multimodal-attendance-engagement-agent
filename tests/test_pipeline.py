"""Automated verification suite for ClassPulse AI.

Tests face detector, WebRTC video/audio processors, ICE server configuration,
audio pipeline, CSV resilience, session persistence, teaching effectiveness,
and PDF report generation.
"""

import csv
import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

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
    compile_live_session_record,
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
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_live_audio:
            tmp_live_path = Path(tmp_live_audio.name)

        try:
            with patch("dashboard.LIVE_AUDIO_FILE", tmp_live_path):
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

                # Verify returned audio frame is muted/silent to prevent speaker feedback
                out_samples = out_frame.to_ndarray()
                self.assertEqual(int(np.max(np.abs(out_samples))), 0)

                # Simulate a second frame past the 0.5-second logging interval
                proc.session_start = time.monotonic() - 1.0
                proc.last_logged = time.monotonic() - 0.6
                proc.recv(frame)
                self.assertGreaterEqual(proc.total_intervals, 1)

                diag = proc.get_diagnostics()
                self.assertIn("current_rms", diag)
                self.assertIn("activity_ratio", diag)
                self.assertGreater(diag["total_intervals"], 0)

                # Verify samples were actually written to the live audio CSV
                rows = read_csv_file(tmp_live_path)
                self.assertGreaterEqual(len(rows), 1)
                self.assertIn("RMS Energy", rows[0])
                self.assertIn("Sound Activity", rows[0])
        finally:
            try:
                os.remove(tmp_live_path)
            except OSError:
                pass

    def test_audio_analysis_with_synthetic_audio(self):
        # Generate 1 second 440Hz sine wave
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        import soundfile as sf
        bio = io.BytesIO()
        sf.write(bio, waveform, sr, format="WAV")
        audio_bytes = bio.getvalue()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_audio:
            tmp_audio_path = Path(tmp_audio.name)

        try:
            with patch("dashboard.AUDIO_FILE", tmp_audio_path):
                summary = run_audio_analysis_on_bytes(audio_bytes, "synthetic_test.wav")
                self.assertGreater(summary["intervals"], 0)
                self.assertGreater(summary["max_rms"], 0.01)
                self.assertGreater(summary["active_intervals"], 0)
                self.assertTrue(tmp_audio_path.exists())
        finally:
            try:
                os.remove(tmp_audio_path)
            except OSError:
                pass

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

            from dashboard import load_session_history, run_video_demonstration

            initial_history = load_session_history()
            initial_count = len(initial_history)

            # Isolate video demo test so synthetic test artifact is not written to persistent session storage
            with patch("dashboard.save_session_record") as mock_save:
                res = run_video_demonstration(tmp_path, DummyProgress(), DummyStatus())
                self.assertIn("samples", res)
                self.assertGreater(res["samples"], 0)
                mock_save.assert_called_once()

            # Verify persistent history was NOT modified by the test
            current_history = load_session_history()
            self.assertEqual(len(current_history), initial_count)
            self.assertFalse(
                any(Path(tmp_path).stem in str(s.get("session_name", "")) for s in current_history),
                "Synthetic test video leaked into persistent session history!",
            )
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_json = Path(tmp_dir) / "test_sessions.json"
            tmp_csv = Path(tmp_dir) / "test_records.csv"
            with patch("dashboard.SESSION_HISTORY_FILE", tmp_json), patch("dashboard.SESSION_CSV_FILE", tmp_csv):
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

    def test_compile_live_session_record_success(self):
        """Test compiling live session metrics from recorded face and audio frames."""
        mock_face_rows = [
            {"Timestamp": "2026-10-24T02:00:01", "Elapsed Seconds": "1.0", "Face Count": "2"},
            {"Timestamp": "2026-10-24T02:00:02", "Elapsed Seconds": "2.0", "Face Count": "3"},
            {"Timestamp": "2026-10-24T02:00:03", "Elapsed Seconds": "3.0", "Face Count": "0"},
            {"Timestamp": "2026-10-24T02:00:04", "Elapsed Seconds": "4.0", "Face Count": "2"},
        ]
        mock_audio_rows = [
            {"Timestamp": "2026-10-24T02:00:01", "Elapsed Seconds": "1.0", "RMS Energy": "0.025", "Sound Activity": "Detected"},
            {"Timestamp": "2026-10-24T02:00:02", "Elapsed Seconds": "2.0", "RMS Energy": "0.005", "Sound Activity": "Low"},
            {"Timestamp": "2026-10-24T02:00:03", "Elapsed Seconds": "3.0", "RMS Energy": "0.030", "Sound Activity": "Detected"},
            {"Timestamp": "2026-10-24T02:00:04", "Elapsed Seconds": "4.0", "RMS Energy": "0.010", "Sound Activity": "Low"},
        ]

        rec = compile_live_session_record(
            face_rows=mock_face_rows,
            audio_rows=mock_audio_rows,
            session_name="Computer Science 101",
        )

        self.assertIsNotNone(rec)
        self.assertTrue(rec["session_id"].startswith("sess_"))
        self.assertEqual(rec["session_name"], "Computer Science 101")
        self.assertEqual(rec["total_samples"], 4)
        self.assertEqual(rec["peak_faces"], 3)
        self.assertEqual(rec["avg_faces"], 1.8)
        self.assertEqual(rec["presence_ratio"], 75.0)  # 3 of 4 frames have face > 0
        self.assertEqual(rec["audio_intervals"], 4)
        self.assertEqual(rec["active_audio_intervals"], 2)
        self.assertEqual(rec["active_audio_ratio"], 50.0)
        self.assertEqual(rec["duration_seconds"], 4.0)
        self.assertIn("effectiveness_score", rec)
        self.assertGreater(rec["effectiveness_score"], 0.0)

    def test_compile_live_session_record_no_data(self):
        """Test that empty samples return None without inventing fake data or scores."""
        res = compile_live_session_record(face_rows=[], audio_rows=[])
        self.assertIsNone(res)

    def test_live_session_archive_and_duplicate_prevention(self):
        """Test archiving live session data and verifying re-archiving updates in place without duplicates."""
        mock_face_rows = [
            {"Timestamp": "2026-10-24T02:00:01", "Elapsed Seconds": "1.0", "Face Count": "1"},
            {"Timestamp": "2026-10-24T02:00:02", "Elapsed Seconds": "2.0", "Face Count": "1"},
        ]
        mock_audio_rows = [
            {"Timestamp": "2026-10-24T02:00:01", "Elapsed Seconds": "1.0", "RMS Energy": "0.020", "Sound Activity": "Detected"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_json = Path(tmp_dir) / "session_history.json"
            tmp_csv = Path(tmp_dir) / "classpulse_session_records.csv"

            with patch("dashboard.SESSION_HISTORY_FILE", tmp_json), patch("dashboard.SESSION_CSV_FILE", tmp_csv):
                # 1. Initial archive
                rec1 = compile_live_session_record(mock_face_rows, mock_audio_rows, session_name="Original Title")
                self.assertIsNotNone(rec1)
                save_ok1 = save_session_record(rec1)
                self.assertTrue(save_ok1)

                history1 = load_session_history()
                self.assertEqual(len(history1), 1)
                self.assertEqual(history1[0]["session_name"], "Original Title")

                # 2. Second archive of same session with updated label
                rec2 = compile_live_session_record(mock_face_rows, mock_audio_rows, session_name="Updated Title")
                self.assertEqual(rec1["session_id"], rec2["session_id"])  # Deterministic session_id
                save_ok2 = save_session_record(rec2)
                self.assertTrue(save_ok2)

                # Verify count is STILL 1 (no duplicate row appended) and title is updated
                history2 = load_session_history()
                self.assertEqual(len(history2), 1)
                self.assertEqual(history2[0]["session_name"], "Updated Title")

    def test_save_session_record_status_and_error_handling(self):
        """Test that save_session_record returns True on success and False on write errors without crashing."""
        sample_rec = {
            "session_id": "test_stat_001",
            "session_name": "Status Test",
            "timestamp": "2026-10-24 02:00",
            "duration_seconds": 10.0,
            "total_samples": 5,
            "avg_faces": 1.0,
            "peak_faces": 1,
            "presence_ratio": 100.0,
            "audio_intervals": 2,
            "active_audio_intervals": 1,
            "active_audio_ratio": 50.0,
            "avg_rms": 0.02,
            "max_rms": 0.04,
            "data_source": "Unit Test",
            "effectiveness_score": 75.0,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_json = Path(tmp_dir) / "session_history.json"
            tmp_csv = Path(tmp_dir) / "classpulse_session_records.csv"

            with patch("dashboard.SESSION_HISTORY_FILE", tmp_json), patch("dashboard.SESSION_CSV_FILE", tmp_csv):
                # Success case
                success = save_session_record(sample_rec)
                self.assertTrue(success)

                # Failure case (e.g. disk/write error)
                with patch("json.dump", side_effect=OSError("Disk write error")):
                    failed = save_session_record(sample_rec)
                    self.assertFalse(failed)


# Backward compatibility alias
TestEduSensePipeline = TestClassPulsePipeline

if __name__ == "__main__":
    unittest.main()
