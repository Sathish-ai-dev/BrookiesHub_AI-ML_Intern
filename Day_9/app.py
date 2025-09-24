# app.py
import streamlit as st
import cv2
from detect import detect_objects

st.title("🔍 Real-Time Object Detection")

run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break
        frame = detect_objects(frame)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
else:
    st.warning("Camera is off.")
