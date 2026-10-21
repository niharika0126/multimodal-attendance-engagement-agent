
import librosa
import numpy as np
import csv
import os

audio_path = "output/sample_audio.wav"
output_path = "output/audio_activity.csv"

if not os.path.exists(audio_path):
    print("Audio file not found. Extract audio first.")
    raise SystemExit

audio, sample_rate = librosa.load(audio_path, sr=None)

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
        "RMS Energy": round(rms, 6)
    })

# Use a relative threshold to estimate sound activity
max_energy = max(row["RMS Energy"] for row in results)
threshold = max(0.01, max_energy * 0.10)

with open(output_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Timestamp (seconds)",
            "RMS Energy",
            "Sound Activity"
        ]
    )
    writer.writeheader()

    for row in results:
        writer.writerow({
            **row,
            "Sound Activity": (
                "Detected"
                if row["RMS Energy"] >= threshold
                else "Low"
            )
        })

active = sum(row["RMS Energy"] >= threshold for row in results)

print("\n===== AUDIO ACTIVITY REPORT =====")
print(f"Audio duration: {len(audio) / sample_rate:.2f} seconds")
print(f"Intervals analyzed: {len(results)}")
print(f"Intervals with sound activity: {active}")
print(f"Intervals with low sound energy: {len(results) - active}")
print(f"Results saved to: {output_path}")
print("\nNote: Sound energy does not confirm speech or engagement.")
