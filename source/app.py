from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet
import pickle
from datetime import datetime

app = Flask(__name__)

# Load Face Recognition Model
embedder = FaceNet()
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Directories
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DETECTED_FACES_FOLDER = os.path.join(BASE_PATH, "detected_faces")
TIME_DATA_FOLDER = os.path.join(BASE_PATH, "time_data")
EMBEDDINGS_FILE = os.path.join(BASE_PATH, "embeddings.pkl")

os.makedirs(DETECTED_FACES_FOLDER, exist_ok=True)
os.makedirs(TIME_DATA_FOLDER, exist_ok=True)

# Load known faces embeddings
with open(EMBEDDINGS_FILE, "rb") as f:
    known_faces = pickle.load(f)

# Function to extract faces
def extract_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)
    faces = []

    if results.detections:
        print(f"Detected {len(results.detections)} face(s)")  # Debugging print
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x, y, width, height = (int(bboxC.xmin * w), int(bboxC.ymin * h), 
                                   int(bboxC.width * w), int(bboxC.height * h))
            face = img_rgb[y:y + height, x:x + width]
            if face.shape[0] > 0 and face.shape[1] > 0:
                faces.append((face, (x, y, width, height)))
    else:
        print("No face detected")  # Debugging print
    return faces

# Function to recognize faces
def recognize_face(face_embedding):
    face_embedding = face_embedding / np.linalg.norm(face_embedding)
    min_dist = float("inf")
    name = "Unknown"

    for person, embeddings in known_faces.items():
        for saved_embedding in embeddings:
            dist = np.linalg.norm(face_embedding - saved_embedding)
            print(f"Comparing with {person}: Distance = {dist}")  # Debugging print
            if dist < 0.7 and dist < min_dist:  # Adjusted threshold
                min_dist = dist
                name = person

    print(f"Recognized as: {name}")  # Debugging print
    return name

# Video Streaming Generator
def generate_frames():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Unable to access webcam.")  # Debugging print
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame.")  # Debugging print
            break
        frame = cv2.flip(frame, 1)

        # Face Recognition
        faces = extract_face(frame)
        for face, (x, y, width, height) in faces:
            face_resized = cv2.resize(face, (160, 160))
            face_embedding = embedder.embeddings([face_resized])[0]
            name = recognize_face(face_embedding)

            # Draw rectangle and name on the frame
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # **Save detected faces**
            if name != "Unknown":  # Save only recognized faces
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Unique timestamp
                face_filename = f"{name}_{timestamp}.jpg"
                face_path = os.path.join(DETECTED_FACES_FOLDER, face_filename)
                cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))  # Convert back to BGR before saving
                print(f"Saved detected face: {face_path}")  # Debugging print

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

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/new-crim')
def newcrim():
    return render_template('newcrim.html')

if __name__ == "__main__":
    app.run(debug=True)
