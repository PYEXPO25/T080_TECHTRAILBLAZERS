import cv2
import numpy as np
import os
import datetime
import mediapipe as mp
from keras_facenet import FaceNet

# Initialize FaceNet & MediaPipe Face Detection
embedder = FaceNet()
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

# Paths
BASE_PATH = r"E:\Hackathon\source"
DATASET_PATH = os.path.join(BASE_PATH, "dataset")
DETECTED_FACES_FOLDER = os.path.join(BASE_PATH, "detected_faces")
TIME_DATA_FOLDER = os.path.join(BASE_PATH, "time_data")

# Create folders if they do not exist
os.makedirs(DETECTED_FACES_FOLDER, exist_ok=True)
os.makedirs(TIME_DATA_FOLDER, exist_ok=True)

# Function to extract face from image using MediaPipe
def extract_face(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)

    if results.detections:
        faces = []
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            face = img_rgb[y:y + height, x:x + width]
            if face.shape[0] > 0 and face.shape[1] > 0:
                faces.append(face)

        return faces
    return None

# Function to create embeddings with multiple faces per person
def create_embeddings():
    known_faces = {}

    for person_name in os.listdir(DATASET_PATH):
        person_folder = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_folder):
            continue

        embeddings = []
        img_list = os.listdir(person_folder)[:100]  # Limit to 100 images per person

        for img_name in img_list:
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            faces = extract_face(img)
            if not faces:
                print(f"[ERROR] No face found in {img_path}")
                continue

            for face in faces:
                face = cv2.resize(face, (160, 160))  # Resize to 160x160
                raw_embedding = embedder.embeddings([face])[0]
                normalized_embedding = raw_embedding / np.linalg.norm(raw_embedding)
                embeddings.append(normalized_embedding)

        if embeddings:
            known_faces[person_name] = np.array(embeddings)  # Store as numpy array
            print(f"[INFO] {person_name} embeddings saved.")

    return known_faces

# Load known face embeddings
known_faces = create_embeddings()
print(f"Loaded embeddings for: {list(known_faces.keys())}")

# Function to recognize a face
def recognize_face(face_embedding):
    face_embedding = face_embedding / np.linalg.norm(face_embedding)  # Normalize input
    min_dist = float("inf")
    name = "Unknown"

    for person, embeddings in known_faces.items():
        for saved_embedding in embeddings:
            dist = np.linalg.norm(face_embedding - saved_embedding)

            if dist < 0.6 and dist < min_dist:  # Lower threshold for accuracy
                min_dist = dist
                name = person

    return name  # Return "Unknown" only if no match found


# Function to save detected face
# Function to save detected face with full image and bounding box
def save_detected_face(full_frame, x, y, width, height, name):
    # Draw bounding box around the detected face
    cv2.rectangle(full_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(full_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    # Save the full image with bounding box inside detected_faces folder
    image_filename = os.path.join(DETECTED_FACES_FOLDER, f"{name}_{timestamp}.jpg")
    cv2.imwrite(image_filename, full_frame)
    print(f"[INFO] Saved Full Image: {image_filename}")

    # Save the timestamp in a text file inside time_data folder
    time_filename = os.path.join(TIME_DATA_FOLDER, f"{name}.txt")
    with open(time_filename, "a") as file:  # Append mode
        file.write(f"{timestamp}\n")
    print(f"[INFO] Logged Time for {name}: {timestamp}")


# Ask user for input choice
print("Choose an option:")
print("1. Use Laptop Camera (Live Detection)")
print("2. Use MP4 Video File for Detection")
choice = input("Enter 1 or 2: ")

# Choose source based on input
if choice == "1":
    cap = cv2.VideoCapture(0)  # Webcam
elif choice == "2":
    VIDEO_PATH = input("Enter the path to the video file: ")
    cap = cv2.VideoCapture(VIDEO_PATH)  # MP4 File
else:
    print("Invalid choice! Exiting...")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame to mirror it (right to left)
    frame = cv2.flip(frame, 1)  # 1 means horizontal flip

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            face = img_rgb[y:y + height, x:x + width]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue  # Skip empty detections

            # Get face embedding
            face_embedding = embedder.embeddings([face])[0]
            face_embedding = face_embedding / np.linalg.norm(face_embedding)  # Normalize

            # Recognize face
            name = recognize_face(face_embedding)

            # Save detected face if recognized
            if name != "Unknown":
                save_detected_face(frame, x, y, width, height, name)

            # Draw bounding box & name
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
