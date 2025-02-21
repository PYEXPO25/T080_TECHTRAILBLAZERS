import cv2
import dlib
import os

# Load dlib's face detector
detector = dlib.get_frontal_face_detector()

# Get user input
name = input("Enter the name of the person in the video: ")
video_path = input("Enter the path of the video: ")

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Unable to open video file.")
    exit()

# Get total frame count
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define step size to extract exactly 100 images
step = max(1, total_frames // 100)

# Create dataset directory
output_dir = f"dataset/{name}"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0
saved_images = 0

while saved_images < 100:  # Stop after saving 100 images
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Process frames at calculated step interval
    if frame_count % step != 0:
        frame_count += 1
        continue

    frame_count += 1

    # Convert frame to grayscale for better face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    if len(faces) == 0:
        continue  # Skip frames where no face is detected

    # Assume the first detected face is the main face
    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Expand the bounding box slightly
    padding_x = int(w * 0.2)  # Expand 20% horizontally
    padding_y = int(h * 0.3)  # Expand 30% vertically
    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    w = min(frame.shape[1] - x, w + 2 * padding_x)
    h = min(frame.shape[0] - y, h + 2 * padding_y)

    # Crop the expanded face
    cropped_face = frame[y:y+h, x:x+w]

    # Resize to 200x200 (better resolution for face recognition)
    clear_face = cv2.resize(cropped_face, (200, 200))

    # Apply contrast enhancement (helps with face clarity)
    clear_face = cv2.convertScaleAbs(clear_face, alpha=1.2, beta=30)

    # Save the cropped face
    image_path = os.path.join(output_dir, f"face_{saved_images}.jpg")
    cv2.imwrite(image_path, clear_face)

    saved_images += 1

cap.release()
print(f"✅ Extracted exactly 100 **clear** face images from video. Saved in: {output_dir}")
