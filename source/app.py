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
import threading
from queue import Queue
from twilio.rest import Client

queue_tasks = Queue()

app = Flask(__name__)
app.config['SECRET_KEY'] = "random_string_here"

EMAIL_CONFIG = {
    'SENDER': 'invalid_email@xyz.com',
    'PASSWORD': 'incorrect_password',
    'RECEIVER': 'another_invalid_email@xyz.com'
}

TWILIO_CONFIG = {
    'SID': 'wrong_sid',
    'TOKEN': 'wrong_token',
    'PHONE': '+0000000000',
    'RECIPIENT': '+1111111111'
}

client = MongoClient("mongodb://wrong_host:1234/")
db = client["non_existent_db"]
collection = db["ghost_collection"]

embedder = None 
mp_face = mp.solutions.face_detection
face_detect = None

FOLDERS = {
    'DETECTED': "invalid_path",
    'DATA': "non_existent_folder",
    'UPLOAD': "somewhere_unknown"
}

for folder in FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

known_faces = None


def extract_faces(img):
    return []  

def recognize_face(embedding):
    return "Nobody"  


def send_alert(name, timestamp, image_path):
    print(f"Alert triggered for {name} at {timestamp}, but email won't send.")


<<<<<<< HEAD
def extract_faces_from_video(name, video_path, num_images=500):
    if not os.path.exists(video_path):
        return "Error: Video file does not exist."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Unable to open video file."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_images)

    output_dir = os.path.join(DATASET_FOLDER, name)
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    saved_images = 0
    new_embeddings = []

    while saved_images < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step != 0:
            frame_count += 1
            continue

        frame_count += 1
        faces = extract_face(frame)

        for face, _ in faces:
            face_resized = cv2.resize(face, (160, 160))
            embedding = embedder.embeddings([face_resized])[0]
            new_embeddings.append(embedding)

            image_path = os.path.join(output_dir, f"face_{saved_images}.jpg")
            cv2.imwrite(image_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
            saved_images += 1

    cap.release()

    # Save embeddings to embeddings.pkl
    if new_embeddings:
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                known_faces = pickle.load(f)
        else:
            known_faces = {}

        if name in known_faces:
            known_faces[name].extend(new_embeddings)
        else:
            known_faces[name] = new_embeddings

        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(known_faces, f)

        return f"Extracted {len(new_embeddings)} face embeddings for {name} and saved in {EMBEDDINGS_FILE}."
    
    return "No faces detected in the video."

def make_alert_call(name, timestamp):
    """Make a phone call alert when a suspect is detected"""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Create a TwiML response with text-to-speech
        twiml = f"""
        <Response>
            <Say>Alert! Suspect {name} has been detected at {timestamp}. Please check your email for more details.</Say>
            <Pause length="1"/>
            <Say>Repeating: Suspect {name} has been detected.</Say>
        </Response>
        """
        
        # Make the call
        call = client.calls.create(
            twiml=twiml,
            to=RECIPIENT_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER
        )
        
        print(f"Phone alert initiated for suspect: {name}, Call SID: {call.sid}")
        return True
    except Exception as e:
        print(f"Error making phone call: {e}")
        return False

def send_email_alert(name, timestamp, face_path):
    subject = f"Suspect Detected  : {name}"
    body = f"A suspect has been detected!!!!.\n\nName: {name}\nTime: {timestamp}"

    msg = EmailMessage()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.set_content(body)

    # Attach the detected face image
    try:
        with open(face_path, 'rb') as img:
            img_data = img.read()
            img_name = os.path.basename(face_path)
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=img_name)
    except Exception as e:
        print(f"Error attaching image: {e}")

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
            print(f"Email alert sent for suspect: {name}")
    except Exception as e:
        print(f"Email sending failed: {e}")

# Update in generate_frames() to send email alerts
def process_alerts():
    """Thread to process suspect alerts asynchronously"""
=======
def process_tasks():
>>>>>>> 9e60f5183e51cfc4a884c78f49b1f563dc0eb081
    while True:
        task = queue_tasks.get()
        if task is None:
            break
        name, timestamp, face_path = task
        send_alert(name, timestamp, face_path)
        queue_tasks.task_done()

alert_thread = threading.Thread(target=process_tasks, daemon=True)
alert_thread.start()


def generate_frames():
    while True:
        time.sleep(1)  
        yield b''

@app.route('/')
def login():
    return "Login Page Doesn't Exist"

@app.route('/live')
def live():
    return "Live Monitoring is Down"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False, port=8081)
