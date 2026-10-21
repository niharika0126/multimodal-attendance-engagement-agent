
import cv2
import mediapipe as mp
import csv
import os
import time
from datetime import datetime

model_path = "models/face_detector.tflite"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Face detector model not found: {model_path}")

# 0 selects the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam. Check camera permissions.")
    raise SystemExit

options = mp.tasks.vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=model_path
    ),
    min_detection_confidence=0.5
)

csv_path = os.path.join(output_folder, "live_face_presence.csv")
start_time = time.time()
frame_number = 0

try:
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp",
            "Elapsed Seconds",
            "Face Count"
        ])

        with mp.tasks.vision.FaceDetector.create_from_options(
            options
        ) as detector:

            print("Live webcam started. Press Q to quit.")

            while True:
                success, frame = cap.read()

                if not success:
                    print("Could not read webcam frame.")
                    break

                frame_number += 1
                elapsed = time.time() - start_time

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                results = detector.detect(mp_image)
                face_count = len(results.detections)
                height, width = frame.shape[:2]

                for detection in results.detections:
                    box = detection.bounding_box
                    x1 = max(0, box.origin_x)
                    y1 = max(0, box.origin_y)
                    x2 = min(width, x1 + box.width)
                    y2 = min(height, y1 + box.height)

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (16, 220, 140), 2
                    )

                    score = detection.categories[0].score
                    cv2.putText(
                        frame,
                        f"Face {score:.0%}",
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (16, 220, 140),
                        2
                    )

                cv2.putText(
                    frame,
                    f"Faces detected: {face_count}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (16, 220, 140),
                    2
                )

                cv2.putText(
                    frame,
                    f"Session: {int(elapsed)} sec",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    round(elapsed, 2),
                    face_count
                ])
                file.flush()

                cv2.imshow("EduSense AI - Live Face Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

finally:
    cap.release()
    cv2.destroyAllWindows()

print(f"Live face data saved to {csv_path}")
