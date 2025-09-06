import cv2
import os
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

video_folder = "D:/ResQDroneAI/videoss"
model = YOLO("yolov8s.pt")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = frame.copy()

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_crop = frame[y1:y2, x1:x2]

                    # Pose estimation
                    person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    pose_result = pose.process(person_rgb)

                    label = "Unknown"
                    color = (0, 255, 0)

                    if pose_result.pose_landmarks:
                        landmarks = pose_result.pose_landmarks.landmark

                        # Get y values of shoulder and hip to estimate vertical span
                        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
                        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y

                        vertical_span = abs((left_shoulder + right_shoulder)/2 - (left_hip + right_hip)/2)

                        if vertical_span < 0.2:
                            label = "URGENT - Lying"
                            color = (0, 0, 255)
                        else:
                            label = "Standing"
                            color = (0, 255, 0)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 🔥 Simulate Thermal View (right side)
        thermal = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
        combined = np.hstack((annotated_frame, thermal))

        cv2.imshow("ResQDrone AI - Posture & Thermal", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()