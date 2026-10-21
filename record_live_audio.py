
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

OUTPUT_FILE = Path("output/live_audio_activity.csv")
SAMPLE_RATE = 16000
WINDOW_SECONDS = 0.5

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Live microphone recording started.")
print("Speak normally. Press Ctrl+C to stop.")

try:
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp",
            "Elapsed Seconds",
            "RMS Energy",
            "Sound Activity",
        ])

        start_time = time.monotonic()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * WINDOW_SECONDS),
        ) as stream:
            while True:
                audio, overflowed = stream.read(
                    int(SAMPLE_RATE * WINDOW_SECONDS)
                )

                rms = float(np.sqrt(np.mean(audio ** 2)))
                elapsed = time.monotonic() - start_time

                # Provisional threshold; calibrate for your microphone.
                activity = "Detected" if rms >= 0.01 else "Low"

                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    round(elapsed, 2),
                    round(rms, 6),
                    activity,
                ])
                file.flush()

                print(
                    f"{elapsed:6.1f}s | RMS={rms:.6f} "
                    f"| {activity}"
                    + (" | Audio overflow" if overflowed else "")
                )

except KeyboardInterrupt:
    print(f"\nRecording stopped. CSV saved to: {OUTPUT_FILE}")

except Exception as error:
    print(f"\nAudio recording failed: {error}")
