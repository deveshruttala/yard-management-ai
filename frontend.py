import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path

# Load YOLO model
def load_yolo_model(weights_path, config_path, classes_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(classes_path, "r") as f:
        classes = f.read().strip().split("\n")
    return net, classes

# Function to process video and extract number plate
def process_video(video_path, yolo_net, yolo_classes, conf_threshold=0.5, nms_threshold=0.4):
    cap = cv2.VideoCapture(video_path)
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    number_plate_text = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        layer_outputs = yolo_net.forward(yolo_net.getUnconnectedOutLayersNames())

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf_threshold:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            label = str(yolo_classes[class_ids[i]])

            if label == "number_plate":  # Assuming the class name for the number plate is "number_plate"
                number_plate_text = "Extracted number plate text"  # Add OCR logic to extract the number plate text

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    return output_video_path, number_plate_text

# Streamlit app
st.title("Truck Video Processing with YOLO")

# Load YOLO model
weights_path = "yolov3.weights"  # Replace with your YOLO weights path
config_path = "yolov3.cfg"  # Replace with your YOLO config path
classes_path = "coco.names"  # Replace with your YOLO classes path

yolo_net, yolo_classes = load_yolo_model(weights_path, config_path, classes_path)

uploaded_video = st.file_uploader("Upload a truck video", type=["mp4", "avi", "mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.write("Processing video...")
    output_video_path, number_plate_text = process_video(video_path, yolo_net, yolo_classes)

    st.write("Number Plate Text:")
    st.write(number_plate_text)

    st.write("Processed Video:")
    st.video(output_video_path)
