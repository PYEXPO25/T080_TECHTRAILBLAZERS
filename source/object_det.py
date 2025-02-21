from ultralytics import YOLO
import cv2

# Load trained YOLO model
model = YOLO(r"C:\Users\vinuk\OneDrive\Desktop\PYEXPO\T080_TECHTRAILBLAZERS\source\model\best.pt")  # Update path if needed

# Open webcam
cap = cv2.VideoCapture(0)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection
    results = model(frame)

    # Display results
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cls = int(box.cls[0])  # Class ID
            conf = box.conf[0]  # Confidence score
            
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Live Object Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
