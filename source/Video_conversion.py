import cv2
import dlib
import os

def extract_faces_from_video(name, video_path, num_images=100):
    # Ensure the path is valid
    if not os.path.exists(video_path):
        print("Error: Video file does not exist.")
        return

    # Load dlib's face detector
    detector = dlib.get_frontal_face_detector()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file. Check file path and format.")
        return

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define step size to extract exactly `num_images` images
    step = max(1, total_frames // num_images)

    # Create dataset directory
    output_dir = os.path.join("source//dataset", name)
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    saved_images = 0

    while saved_images < num_images:
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

        # Resize to 200x200
        clear_face = cv2.resize(cropped_face, (200, 200))

        # Apply contrast enhancement
        clear_face = cv2.convertScaleAbs(clear_face, alpha=1.2, beta=30)

        # Save the cropped face
        +        image_path = os.path.join(output_dir, f"face_{saved_images}.jpg")
        cv2.imwrite(image_path, clear_face)

        saved_images += 1

    cap.release()
    print(f"Extracted {num_images} face images from video. Saved in: {output_dir}")

# Get user input
name = input("Enter the name of the person in the video: ")
video_path = input("Enter the path of the video: ").strip('\"')  # Remove quotes if pasted

# Call function
extract_faces_from_video(name, video_path)
