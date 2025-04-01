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


def process_tasks():
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
