"""Automated verification suite for EduSense AI.

Tests face detector, WebRTC processor, ICE server configuration,
audio pipeline, CSV resilience, and empty state handling.
"""

import csv
import io
import tempfile
import unittest
from pathlib import Path

import av
import cv2
import numpy as np

from dashboard import (
    BASE_DIR,
    MODEL_PATH,
    FaceDetectionProcessor,
    get_ice_servers,
    get_rtc_configuration,
    number,
    read_csv_file,
    run_audio_analysis_on_bytes,
)


class TestEduSensePipeline(unittest.TestCase):
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
        servers = get_ice_servers()
        self.assertIsInstance(servers, list)
        self.assertGreater(len(servers), 0)

        # Confirm STUN is present
        stun_found = any(
            any("stun:" in str(url) for url in s.get("urls", []))
            for s in servers
        )
        self.assertTrue(stun_found, "At least one STUN server must be configured.")

        # Confirm TURN relay is present (for Cloudflare / Streamlit Cloud / NAT bypass)
        turn_found = any(
            any("turn:" in str(url) for url in s.get("urls", []))
            for s in servers
        )
        self.assertTrue(
            turn_found,
            "At least one TURN relay server must be configured for cloud NAT bypass.",
        )

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
                import os
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
