import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "output" / "face_presence.csv"

if not csv_path.exists():
    print(f"Error: {csv_path} not found. Run face_detection.py or the dashboard demonstration first.")
    raise SystemExit(1)

with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

if not rows:
    print("No frame data found in the CSV.")
    raise SystemExit(1)

total_frames = len(rows)
frames_with_face = sum(
    row.get("Face Detected", "").strip().lower() == "true"
    for row in rows
)
frames_without_face = total_frames - frames_with_face
face_percentage = (frames_with_face / total_frames) * 100 if total_frames else 0.0

try:
    last_timestamp = float(rows[-1].get("Timestamp (seconds)", 0.0))
except (ValueError, TypeError):
    last_timestamp = 0.0

print("\n===== CLASSROOM FACE PRESENCE REPORT =====")
print(f"Total frames processed: {total_frames}")
print(f"Frames with a detected face: {frames_with_face}")
print(f"Frames without a detected face: {frames_without_face}")
print(f"Face-presence percentage: {face_percentage:.2f}%")
print(f"Approximate last frame timestamp: {last_timestamp:.2f} seconds")
print("\nNote: Face presence alone does not confirm attendance or engagement.")
