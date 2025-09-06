import streamlit as st
import cv2
import os
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from streamlit_lottie import st_lottie
import json
import base64

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="ResQDroneAI - Emergency Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ LOAD LOTTIE FUNCTION ------------------
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

# ------------------ LOAD MODELS ------------------
model = YOLO("yolov8s.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# ------------------ SIDEBAR NAVIGATION ------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "About Project"])

# ------------------ ABOUT PAGE ------------------
if page == "About Project":
    st.title("📖 About ResQDroneAI")
    st.write("""
    **ResQDroneAI** is an AI-powered emergency detection system designed for drones.

    It combines **YOLOv8 object detection** with **MediaPipe Pose Estimation** to identify if people in drone footage are **standing or lying down** — potentially indicating a disaster or medical emergency.

    If someone is lying flat, the system raises an **emergency alert** and plays a warning audio.
    """)
    st.subheader("🎞️ How It Works")
    try:
        st_lottie(load_lottie("animation.json"), height=300)
    except:
        st.warning("⚠️ Animation file not found. Please place `animation.json` in the root directory.")
    st.stop()

# ------------------ HOME PAGE ------------------
st.title("🚁 ResQDroneAI - Emergency Detection from Drone Footage")
st.markdown("Upload drone-captured video footage to detect human postures and potential emergencies.")

# ------------------ AUDIO ALERT SETUP ------------------
audio_html = None
if os.path.exists("emergency.mp3"):
    with open("emergency.mp3", "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode()
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
        """

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📂 Upload a drone video (MP4 format)", type=["mp4"])

if uploaded_file is not None:
    # Save uploaded file
    os.makedirs("uploaded_videos", exist_ok=True)
    video_path = os.path.join("uploaded_videos", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    # Display initial processing message
    status_placeholder = st.empty()
    status_placeholder.success("✅ Video uploaded successfully. Processing started...")

    # ------------------ INITIALIZE SESSION STATE ------------------
    if "urgent_triggered" not in st.session_state:
        st.session_state.urgent_triggered = False

    urgent_detected = False
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = frame.copy()

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # class 0 = person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0:
                        continue

                    person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    pose_result = pose.process(person_rgb)

                    label = "Unknown"
                    color = (0, 255, 0)

                    if pose_result.pose_landmarks:
                        landmarks = pose_result.pose_landmarks.landmark
                        l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                        r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
                        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y

                        vertical_span = abs((l_sh + r_sh)/2 - (l_hip + r_hip)/2)

                        if vertical_span < 0.2:
                            label = "🚨 URGENT - Lying"
                            color = (0, 0, 255)
                            urgent_detected = True

                            if not st.session_state.urgent_triggered:
                                st.session_state.urgent_triggered = True
                                st.error("🚨 EMERGENCY DETECTED: Someone appears to be lying down!")
                                if audio_html:
                                    st.markdown(audio_html, unsafe_allow_html=True)

                        else:
                            label = "Standing"
                            color = (0, 255, 0)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Combine annotated + thermal
        thermal = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
        combined = np.hstack((annotated_frame, thermal))

        # Display frame
        stframe.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB),
                      channels="RGB", use_container_width=True)

    cap.release()
    pose.close()

    # Final message if no emergency was detected
    if not urgent_detected and not st.session_state.urgent_triggered:
        st.success("✅ No emergency detected in the video.")
