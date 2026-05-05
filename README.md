# AI-Based Pothole Severity Assessment System

1. Project Overview
   
This project develops an AI system capable of detecting potholes in road images and videos and assessing their severity using instance segmentation. The system is built using the YOLOv8 segmentation model and deployed through a Streamlit-based web application that allows users to upload images or videos for pothole detection.
The application performs pothole detection, severity estimation based on area coverage, risk analysis, and generates downloadable reports for infrastructure monitoring.
The goal of the system is to support smart road monitoring and assist municipal authorities in prioritizing road maintenance.
________________________________________
Business Problem
Manual road inspection is:

Time-consuming
Expensive
Inconsistent
This system automates pothole detection using AI.
________________________________________
💡 Solution
YOLOv8 object detection model
Image augmentation (~8700 images)
Real-time detection using Streamlit
________________________________________
🛠 Tech Stack
Python
YOLOv8 (Ultralytics)
OpenCV
NumPy, Pandas
Streamlit
Roboflow
________________________________________
📊 Dataset
Original images: 1,119
After augmentation: ~8,700 images
Annotation format: YOLO
________________________________________
⚙️ Model Details
Model: YOLOv8
Image size: 640×640
Epochs: 50–100
Metrics: Precision, Recall, mAP
________________________________________
📈 Results
Achieved ~0.75–0.85 mAP
Improved small pothole detection using augmentation
________________________________________
🚀 How to Run
pip install -r requirements.txt streamlit run app.py
________________________________________
📂 Project Structure
app.py → Streamlit app
notebooks → training notebooks
docs → project documentation

