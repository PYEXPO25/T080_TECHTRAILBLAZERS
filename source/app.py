from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
import cv2
import os
import numpy as np
import mediapipe as mp
from keras_facenet import FaceNet
import pickle
from datetime import datetime

app = Flask(__name__)
app.secret_key = "your_secret_key"

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
            if dist < 0.7 and dist < min_dist:
                min_dist = dist
                name = person
    return name

# Store alerts globally
alerts = []

def generate_frames():
    global alerts
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    while True:
        success, frame = cap.read()
        if not success:
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

            # Save detected faces & trigger alert
            if name != "Unknown":
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                face_filename = f"{name}_{timestamp}.jpg"
                face_path = os.path.join(DETECTED_FACES_FOLDER, face_filename)
                cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                # Store alert message
                alerts.append({"name": name, "time": timestamp})

        # Encode Frame for Streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

# API to send alerts to frontend
@app.route('/get_alerts')
def get_alerts():
    global alerts
    return jsonify(alerts)


# Function to extract faces from video and store embeddings
def extract_faces_from_video(name, video_path, num_images=100):
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

@app.route('/new-crim', methods=['GET', 'POST'])
def newcrim():
    if request.method == 'POST':
        name = request.form.get("name")
        file = request.files["video"]

        if file and name:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            result = extract_faces_from_video(name, filepath)
            flash(result)
            return redirect(url_for('newcrim'))

    return render_template('newcrim.html')

if __name__ == "__main__":
    app.run(debug=True)
