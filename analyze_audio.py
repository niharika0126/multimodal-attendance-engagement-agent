import csv
from pathlib import Path
import librosa
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
audio_path = BASE_DIR / "output" / "sample_audio.wav"
output_path = BASE_DIR / "output" / "audio_activity.csv"

if not audio_path.exists():
    print(f"Audio file not found at: {audio_path}. Place an audio recording or extract audio first.")
    raise SystemExit(1)

audio, sample_rate = librosa.load(str(audio_path), sr=None)

# Calculate sound energy in 0.5-second intervals
window_size = int(sample_rate * 0.5)
results = []

for start in range(0, len(audio), window_size):
    segment = audio[start:start + window_size]
    if len(segment) == 0:
        continue

    rms = float(np.sqrt(np.mean(segment ** 2)))
    timestamp = start / sample_rate

    results.append({
        "Timestamp (seconds)": round(timestamp, 2),
        "RMS Energy": round(rms, 6),
    })

if not results:
    print("No audio segments analyzed.")
    raise SystemExit(1)

# Use a relative threshold to estimate sound activity
max_energy = max(row["RMS Energy"] for row in results)
threshold = max(0.01, max_energy * 0.10)

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Timestamp (seconds)",
            "RMS Energy",
            "Sound Activity",
        ],
    )
    writer.writeheader()

    for row in results:
        writer.writerow({
            **row,
            "Sound Activity": (
                "Detected"
                if row["RMS Energy"] >= threshold
                else "Low"
            ),
        })

active = sum(row["RMS Energy"] >= threshold for row in results)

print("\n===== AUDIO ACTIVITY REPORT =====")
print(f"Audio duration: {len(audio) / sample_rate:.2f} seconds")
print(f"Intervals analyzed: {len(results)}")
print(f"Intervals with sound activity: {active}")
print(f"Intervals with low sound energy: {len(results) - active}")
print(f"Results saved to: {output_path}")
print("\nNote: Sound energy does not confirm speech or engagement.")
