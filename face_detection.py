
import cv2
import mediapipe as mp

print("Starting face detection...")

video_path = "videos/sample.mp4"
model_path = "models/face_detector.tflite"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    raise SystemExit

options = mp.tasks.vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=model_path
    ),
    min_detection_confidence=0.5
)

with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
    print("Video opened! Face detection started.")

    while True:
        success, frame = cap.read()

        if not success:
            print("Video finished.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = detector.detect(mp_image)

        if results.detections:
            for detection in results.detections:
                box = detection.bounding_box

                x = box.origin_x
                y = box.origin_y
                w = box.width
                h = box.height

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

            print("Faces detected:", len(results.detections))

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
