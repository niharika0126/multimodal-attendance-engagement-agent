
import csv
import os

csv_path = os.path.join("output", "face_presence.csv")

if not os.path.exists(csv_path):
    print("Error: face_presence.csv not found. Run face_detection.py first.")
    raise SystemExit

with open(csv_path, "r", newline="") as file:
    rows = list(csv.DictReader(file))

if not rows:
    print("No frame data found in the CSV.")
    raise SystemExit

total_frames = len(rows)
frames_with_face = sum(
    row["Face Detected"].strip().lower() == "true"
    for row in rows
)
frames_without_face = total_frames - frames_with_face

face_percentage = (frames_with_face / total_frames) * 100

last_timestamp = float(rows[-1]["Timestamp (seconds)"])
fps_duration = last_timestamp

print("\n===== CLASSROOM FACE PRESENCE REPORT =====")
print(f"Total frames processed: {total_frames}")
print(f"Frames with a detected face: {frames_with_face}")
print(f"Frames without a detected face: {frames_without_face}")
print(f"Face-presence percentage: {face_percentage:.2f}%")
print(f"Approximate last frame timestamp: {fps_duration:.2f} seconds")
print("\nNote: Face presence alone does not confirm attendance or engagement.")
