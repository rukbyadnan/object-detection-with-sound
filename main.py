import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import threading
import time
import os
import csv
from datetime import datetime
import pygetwindow as gw

# Load YOLO model
weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"
coco_names_path = "coco.names"

net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class labels from coco.names
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Generate random colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Logging: Initialize log file
log_file = "detection_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Object"])

# Access webcam
cap = cv2.VideoCapture(0) # 0 default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Error: Cannot access the webcam")
    exit()

# Minimum time delay between voice announcements (20 seconds)
DETECTION_DELAY = 20
last_detection_time = {}

# Function to generate and play voice in a separate thread
def threaded_speak(text):
    def play_voice():
        try:
            filename = f"deteksi_{int(time.time())}.mp3"
            tts = gTTS(text, lang='id')
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print(f"Error saat memutar suara: {e}")
    
    threading.Thread(target=play_voice, daemon=True).start()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam")
        break

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    height, width, channels = frame.shape

    # Define ROI (Region of Interest)
    roi_start_x, roi_start_y = width // 8, height // 8
    roi_end_x, roi_end_y = width * 7 // 8, height * 7 // 8
    roi = frame[roi_start_y:roi_end_y, roi_start_x:roi_end_x]

    # Draw ROI rectangle (visualization)
    cv2.rectangle(frame, (roi_start_x, roi_start_y), (roi_end_x, roi_end_y), (255, 255, 255), 0)

    # Preprocess ROI for YOLO
    blob = cv2.dnn.blobFromImage(roi, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Analyze detection results
    boxes = []
    confidences = []
    class_ids = []
    detected_objects = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]  # Confidence scores for each class
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Save detections with confidence > 0.5
            if confidence > 0.5:
                # Calculate bounding box coordinates relative to ROI
                center_x = int(obj[0] * (roi_end_x - roi_start_x)) + roi_start_x
                center_y = int(obj[1] * (roi_end_y - roi_start_y)) + roi_start_y
                w = int(obj[2] * (roi_end_x - roi_start_x))
                h = int(obj[3] * (roi_end_y - roi_start_y))

                # Calculate top-left coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected_objects.append(classes[class_id])

    # Apply Non-Maximum Suppression (NMS) to avoid duplicate boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Display label and confidence
            text = f"{label}: {round(confidence * 100, 2)}%"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Handle detected objects with delay
    current_time = time.time()
    for obj in set(detected_objects):
        if obj not in last_detection_time or (current_time - last_detection_time[obj]) > DETECTION_DELAY:
            sentence = f"Ini {obj}"  # Generate sentence
            print(f"Deteksi objek: {sentence}")  # Debugging
            threaded_speak(sentence)  # Play sound in thread
            last_detection_time[obj] = current_time  # Update last detection time

            # Log detection to file
            with open(log_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), obj])

    # Display frame
    cv2.imshow("Object Detection - YOLO", frame)

    # Handle Keypress
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"deteksi_{timestamp}.jpg", frame)
        print(f"Frame disimpan sebagai deteksi_{timestamp}.jpg")
    elif key == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
