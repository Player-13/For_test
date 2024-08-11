import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Path to your YOLOv8 model file

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform inference
    results = model(frame)

    # Render results on the frame
    annotated_frame = results.render()[0]

    # Display the frame with detection results
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
