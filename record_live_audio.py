import csv
from datetime import datetime
from pathlib import Path
import time
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "output" / "live_audio_activity.csv"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 0.5

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

try:
    import sounddevice as sd
except Exception as e:
    print(f"Error: sounddevice could not be imported: {e}")
    print("Ensure PortAudio is installed on your operating system.")
    raise SystemExit(1)

# Check for input devices
try:
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    if default_input is None or default_input < 0:
        print("Warning: No default microphone device detected.")
except Exception as e:
    print(f"Warning checking audio devices: {e}")

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

                # Provisional threshold; calibrate for your microphone
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
