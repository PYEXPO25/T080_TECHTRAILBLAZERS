from flask import Flask, render_template, Response
import cv2
import os
import datetime
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet
from ultralytics import YOLO
import pickle

app = Flask(__name__)

# Load YOLO Model for Weapon Detection
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "model", "best.pt")
yolo_model = YOLO(MODEL_PATH)

# Load Face Recognition Model
embedder = FaceNet()
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

# Directories
DETECTED_FACES_FOLDER = os.path.join(BASE_PATH, "detected_faces")
TIME_DATA_FOLDER = os.path.join(BASE_PATH, "time_data")
WEAPON_FOLDER = os.path.join(BASE_PATH, "detected_weapons")
WEAPON_TIME_LOG = os.path.join(BASE_PATH, "weapon_time_data", "weapon_log.txt")
EMBEDDINGS_FILE = os.path.join(BASE_PATH, "embeddings.pkl")

os.makedirs(DETECTED_FACES_FOLDER, exist_ok=True)
os.makedirs(TIME_DATA_FOLDER, exist_ok=True)
os.makedirs(WEAPON_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(WEAPON_TIME_LOG), exist_ok=True)

# Load known faces embeddings
with open(EMBEDDINGS_FILE, "rb") as f:
    known_faces = pickle.load(f)

# Class Labels
CLASS_LABELS = {0: "Person", 1: "Weapon"}

# Function to extract faces
def extract_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x, y, width, height = (int(bboxC.xmin * w), int(bboxC.ymin * h), 
                                   int(bboxC.width * w), int(bboxC.height * h))
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

# Video Streaming Generator
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        # Weapon Detection
        results = yolo_model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = CLASS_LABELS.get(class_id, "Unknown")  
                color = (0, 255, 0) if label == "Person" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if label == "Weapon":
                    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                    weapon_filename = os.path.join(WEAPON_FOLDER, f"weapon_{timestamp}.jpg")
                    weapon_crop = frame[y1:y2, x1:x2]
                    cv2.imwrite(weapon_filename, weapon_crop)
                    with open(WEAPON_TIME_LOG, "a") as file:
                        file.write(f"{timestamp}\n")

        # Face Recognition
        faces = extract_face(frame)
        for face, (x, y, width, height) in faces:
            face_resized = cv2.resize(face, (160, 160))
            face_embedding = embedder.embeddings([face_resized])[0]
            name = recognize_face(face_embedding)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Encode Frame for Streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/live-mon')
def live_mon():
    return render_template("livemon.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
