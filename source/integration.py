import cv2
import torch
import numpy as np
import os
import datetime
import mediapipe as mp
from keras_facenet import FaceNet
from ultralytics import YOLO

# Load YOLOv8 Model
yolo_model = YOLO(r"E:\Hackathon\source\model\best.pt")

# Load Face Recognition Model
embedder = FaceNet()
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

# Directories for detected faces, timestamps, and weapons
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DETECTED_FACES_FOLDER = os.path.join(BASE_PATH, "detected_faces")
TIME_DATA_FOLDER = os.path.join(BASE_PATH, "time_data")
WEAPON_FOLDER = os.path.join(BASE_PATH, "detected_weapons")
WEAPON_TIME_LOG = os.path.join(BASE_PATH, "weapon_time_data", "weapon_log.txt")

os.makedirs(DETECTED_FACES_FOLDER, exist_ok=True)
os.makedirs(TIME_DATA_FOLDER, exist_ok=True)
os.makedirs(WEAPON_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(WEAPON_TIME_LOG), exist_ok=True)

# Class Labels (Ensure they match your YOLO model's output)
CLASS_LABELS = {0: "Person", 1: "Weapon"}  # Modify if your model has different indices

# Load known faces embeddings
DATASET_PATH = os.path.join(BASE_PATH, "dataset")

# Keep track of saved faces
saved_faces = set()

def create_embeddings():
    known_faces = {}
    for person_name in os.listdir(DATASET_PATH):
        person_folder = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_folder):
            continue
        embeddings = []
        for img_name in os.listdir(person_folder)[:100]:  # Limit to 100 images per person
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = extract_face(img)
            if not faces:
                continue
            for face, _ in faces:
                face = cv2.resize(face, (160, 160))
                raw_embedding = embedder.embeddings([face])[0]
                normalized_embedding = raw_embedding / np.linalg.norm(raw_embedding)
                embeddings.append(normalized_embedding)
        if embeddings:
            known_faces[person_name] = np.array(embeddings)
    return known_faces

# Function to extract faces
def extract_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            face = img_rgb[y:y + height, x:x + width]
            if face.shape[0] > 0 and face.shape[1] > 0:
                faces.append((face, (x, y, width, height)))
    return faces

# Function to recognize faces
def recognize_face(face_embedding):
    face_embedding = face_embedding / np.linalg.norm(face_embedding)
    min_dist = float("inf")
    name = "Unknown"
    for person, embeddings in known_faces.items():
        for saved_embedding in embeddings:
            dist = np.linalg.norm(face_embedding - saved_embedding)
            if dist < 0.6 and dist < min_dist:
                min_dist = dist
                name = person
    return name

# Function to save detected faces and timestamps (only once per session)
def save_detected_face(full_frame, x, y, width, height, name):
    if name in saved_faces:
        return  # Skip if the face is already saved

    saved_faces.add(name)  # Mark this face as saved
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    image_filename = os.path.join(DETECTED_FACES_FOLDER, f"{name}.jpg")  # Store only once

    cv2.rectangle(full_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(full_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(image_filename, full_frame)

    with open(os.path.join(TIME_DATA_FOLDER, f"{name}.txt"), "a") as file:
        file.write(f"{timestamp}\n")

# Load known face embeddings
known_faces = create_embeddings()
print(f"Loaded embeddings for: {list(known_faces.keys())}")

# Open Webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror the frame

    # YOLOv8 Object Detection
    results = yolo_model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])  # Get class ID
            confidence = float(box.conf[0])  # Get confidence score

            label = CLASS_LABELS.get(class_id, " ")  # Fetch label or default to "Unknown"
            color = (0, 255, 0) if label == "Person" else (0, 0, 255)  # Green for Person, Red for Weapon

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # If a weapon is detected, save it with a timestamp
            if label == "Weapon":
                timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                weapon_filename = os.path.join(WEAPON_FOLDER, f"weapon_{timestamp}.jpg")

                # Save cropped weapon image
                weapon_crop = frame[y1:y2, x1:x2]
                cv2.imwrite(weapon_filename, weapon_crop)

                # Log timestamp
                with open(WEAPON_TIME_LOG, "a") as file:
                    file.write(f"{timestamp}\n")

    # Face Recognition
    faces = extract_face(frame)
    for face, (x, y, width, height) in faces:
        face_resized = cv2.resize(face, (160, 160))
        face_embedding = embedder.embeddings([face_resized])[0]
        name = recognize_face(face_embedding)
        if name != "Unknown":
            save_detected_face(frame, x, y, width, height, name)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display Video Output
    cv2.imshow("YOLOv8 + Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
