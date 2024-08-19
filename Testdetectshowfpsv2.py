import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt').to('cuda')  # Make sure to load the model on GPU if CUDA is available

# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform inference
    results = model(frame, imgsz=640, half=True)

    # Process the results
    annotated_frame = results[0].plot()  # Plot the results on the frame

    # Display the resulting frame
    cv2.imshow('Webcam YOLO Detection', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()


#Key Changes
#imgsz instead of size: Changed size=320 to imgsz=640 in the model call.
#Model Loading on GPU: Ensure model is loaded with .to('cuda') if CUDA is available.
#Result Plotting: Used results[0].plot() to overlay results on the frame.
