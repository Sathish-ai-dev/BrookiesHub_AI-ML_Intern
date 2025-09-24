# 📦 Real-Time Object Detection with YOLOv5, OpenCV, and Streamlit
# 🔍 Overview
This project demonstrates how to build a real-time object detection web app using a webcam feed. It leverages the YOLOv5 deep learning model for object detection, OpenCV for video capture and rendering, and Streamlit for interactive deployment.

# 🚀 Features
 - Real-time object detection using webcam

 - Bounding boxes with class labels and confidence scores

 - Streamlit-based web interface

 - Easy setup and deployment

 - Modular code structure

# 🧰 Tools & Technologies
 - Python 3.10+

 - YOLOv5 (pre-trained yolov5s.pt)

 - OpenCV

 - Streamlit

 - Torch & torchvision

# 📁 Project Structure
Code
yolov5_streamlit_app/
├── app.py                  # Streamlit interface
├── detect.py               # YOLOv5 detection logic
├── yolov5/                 # Cloned YOLOv5 repo
├── yolov5s.pt              # Pre-trained model
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

# ⚙️ Installation & Setup
1. Clone YOLOv5
bash
git clone https://github.com/ultralytics/yolov5
2. Download Pre-trained Model
Download yolov5s.pt from (https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt) YOLOv5 Releases and place it in the project root.

3. Install Dependencies
bash
pip install -r yolov5/requirements.txt
pip install streamlit opencv-python
⚠️ If you encounter numpy or protobuf version conflicts, downgrade to compatible versions:

bash
pip install numpy==1.26.4 protobuf==3.20.3

# 🧠 How It Works
 * detect.py loads YOLOv5 locally and processes each frame from the webcam.

 * Detected objects are annotated with bounding boxes and labels.

 * app.py runs a Streamlit interface that displays the live feed with detections.

# ▶️ Running the App
  - bash
     * streamlit run app.py
     * Then open your browser to:
     * Code : http://localhost:8501
     
# 📹 Demo Video
 - Include a screen recording showing:

 - Live webcam feed

 - Real-time detection

 - Streamlit interface

 - You can use OBS Studio or Loom for recording.

# 📄 Deliverables
✅ app.py and detect.py source code

✅ README.md (this file)

✅ Video demo or live deployment

✅ requirements.txt

# 🌐 Deployment Options
 - Streamlit Cloud

 - Hugging Face Spaces

 - Render

# 🙌 Credits
   * Ultralytics YOLOv5

   * Streamlit

   * OpenCV