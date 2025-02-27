from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
import cv2
import os
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet
import pickle
from datetime import datetime
import time
from pymongo import MongoClient
from email.message import EmailMessage
import smtplib
import ssl

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Email Configuration
EMAIL_SENDER = 'mourishantonyc@gmail.com'
EMAIL_PASSWORD = 'hmxn wppp myla mhkc'
EMAIL_RECEIVER = 'rajmourishantony@gmail.com'

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition"]
collection = db["suspects"]

# Load Face Recognition Model
embedder = FaceNet()
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Directories
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DETECTED_FACES_FOLDER = os.path.join(BASE_PATH, "detected_faces")
TIME_DATA_FOLDER = os.path.join(BASE_PATH, "time_data")
EMBEDDINGS_FILE = os.path.join(BASE_PATH, "embeddings.pkl")
UPLOAD_FOLDER = os.path.join(BASE_PATH, "uploads")
DATASET_FOLDER = os.path.join(BASE_PATH, "dataset")

os.makedirs(DETECTED_FACES_FOLDER, exist_ok=True)
os.makedirs(TIME_DATA_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Load known faces embeddings
with open(EMBEDDINGS_FILE, "rb") as f:
    known_faces = pickle.load(f)
print("Known Faces:", known_faces.keys())

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
            face = img[y:y + height, x:x + width]  
            if face.shape[0] > 0 and face.shape[1] > 0:
                faces.append((face, (x, y, width, height)))
    return faces

def recognize_face(face_embedding):
    face_embedding = face_embedding / np.linalg.norm(face_embedding)
    min_dist = float("inf")
    name = "Unknown"

    for person, embeddings in known_faces.items():
        for saved_embedding in embeddings:
            dist = np.linalg.norm(face_embedding - saved_embedding)
            if dist < 0.7 and dist < min_dist:
                min_dist = dist
                name = person
    return name

def send_email(suspect_name, suspect_time, suspect_image_path):
    subject = "Suspect Detected Alert"
    body = f"Suspect Name: {suspect_name}\nDetection Time: {suspect_time}\n\nThe suspect's image is attached."

    msg = EmailMessage()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.set_content(body)

    with open(suspect_image_path, 'rb') as img_file:
        img_data = img_file.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename="suspect.jpg")

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("Email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")

alerts = []
last_detection_time = {}

def generate_frames():
    global alerts, last_detection_time
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                known_faces = pickle.load(f)

        faces = extract_face(frame)
        detected_names = set()
        current_time = time.time()

        for face, (x, y, width, height) in faces:
            face_resized = cv2.resize(face, (160, 160))
            face_embedding = embedder.embeddings([face_resized])[0]
            name = recognize_face(face_embedding)

            if name not in detected_names:
                detected_names.add(name)

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if name != "Unknown" and (name not in last_detection_time or current_time - last_detection_time[name] >= 10):
                    last_detection_time[name] = current_time

                    person_folder = os.path.join(DETECTED_FACES_FOLDER, name)
                    os.makedirs(person_folder, exist_ok=True)

                    timestamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
                    face_count = len(os.listdir(person_folder))
                    face_filename = f"img_{face_count + 1}.jpg"
                    face_path = os.path.join(person_folder, face_filename)

                    cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                    time_data_path = os.path.join(TIME_DATA_FOLDER, f"{name}.txt")
                    with open(time_data_path, "a") as f:
                        f.write(f"{timestamp}\n")

                    alerts.append({"name": name, "time": timestamp})

                    with open(face_path, "rb") as img_file:
                        image_data = img_file.read()

                    suspect_data = {
                        "suspect_name": name,
                        "detected_image": image_data,
                        "time": timestamp
                    }
                    collection.insert_one(suspect_data)

                    # Send Email Notification
                    send_email(name, timestamp, face_path)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/get_alerts')
def get_alerts():
    global alerts
    return jsonify(alerts)

@app.route('/suspects')
def get_suspects():
    suspects = list(collection.find({}, {"_id": 0, "suspect_name": 1, "time": 1}))
    return jsonify(suspects)

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

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)
