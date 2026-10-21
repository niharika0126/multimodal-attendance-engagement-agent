
import cv2
import mediapipe as mp
import csv
import os

video_path = "videos/sample.mp4"
model_path = "models/face_detector.tflite"

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
frame_number = 0

options = mp.tasks.vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=model_path
    ),
    min_detection_confidence=0.5
)

csv_path = os.path.join(output_folder, "face_presence.csv")

try:
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Frame",
            "Timestamp (seconds)",
            "Face Detected",
            "Face Count"
        ])

        with mp.tasks.vision.FaceDetector.create_from_options(
            options
        ) as detector:
            print("Video opened! Press Q to quit.")

            while True:
                success, frame = cap.read()

                if not success:
                    print("Video finished.")
                    break

                frame_number += 1
                timestamp = (
                    (frame_number - 1) / fps if fps > 0 else 0
                )

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                results = detector.detect(mp_image)
                face_count = len(results.detections)

                writer.writerow([
                    frame_number,
                    round(timestamp, 2),
                    face_count > 0,
                    face_count
                ])

                h, w, _ = frame.shape

                for detection in results.detections:
                    box = detection.bounding_box

                    x = max(0, box.origin_x)
                    y = max(0, box.origin_y)
                    right = min(w, x + box.width)
                    bottom = min(h, y + box.height)

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (right, bottom),
                        (0, 255, 0),
                        2
                    )

                cv2.imshow("Classroom Face Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

finally:
    cap.release()
    cv2.destroyAllWindows()

print(f"Face presence data saved to {csv_path}")
