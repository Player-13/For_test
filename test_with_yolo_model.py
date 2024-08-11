from ultralytics import YOLO
import cv2
import math
import threading
from PIL import Image, ImageTk
import tkinter as tk
from queue import Queue, Empty
import matplotlib.pyplot as plt
import pickle
import os
import time

# Define a delay interval in seconds
COUNT_UPDATE_INTERVAL = 1.8
DELAY_INTERVAL = 1.8  # Delay before counting updates

def process_frame():
    global stop_flag, counts, count_stack, last_update_time, last_count_update_time

    last_update_time = time.time()  # Initialize the last update time
    last_count_update_time = time.time()  # Initialize the last count update time

    while not stop_flag:
        success, img = cap.read()
        if not success:
            print("Failed to capture image.")
            break

        results = model(img, stream=True)

        # Reset counts for each frame
        current_counts = {"Female": 0, "Male": 0}
        processed_boxes = []

        # Coordinates and drawing on the image
        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Check if the box overlaps with already processed boxes
                new_box = [x1, y1, x2, y2]
                overlap = False
                for processed_box in processed_boxes:
                    if compute_iou(new_box, processed_box) > IOU_THRESHOLD:
                        overlap = True
                        break
                
                if not overlap:
                    processed_boxes.append(new_box)

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # Confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    # Class name
                    cls = int(box.cls[0])
                    class_name = classNames[cls]
                    print("Class name -->", class_name)

                    # Update current counts
                    if class_name in current_counts:
                        current_counts[class_name] += 1

                    # Object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(img, class_name, org, font, fontScale, color, thickness)

        # Check if it's time to update the global counts
        current_time = time.time()
        if current_time - last_update_time >= COUNT_UPDATE_INTERVAL:
            # Only update global counts if 20 seconds have passed
            if current_time - last_count_update_time >= DELAY_INTERVAL:
                # Update global counts and stack
                counts["Female"] += current_counts["Female"]
                counts["Male"] += current_counts["Male"]
                count_stack.append(counts.copy())

                # Limit stack size to the last N frames (e.g., 100 frames)
                if len(count_stack) > 100:
                    count_stack.pop(0)

                # Save counts to file
                with open(COUNTS_FILE, "wb") as f:
                    pickle.dump(counts, f)

                last_count_update_time = current_time  # Update the last count update time

            last_update_time = current_time  # Update the last update time

        # Put the frame in the queue
        frame_queue.put(img)

    cap.release()

# File paths for saving and loading counts
COUNTS_FILE = "counts.pkl"

# Initialize webcam
cap = cv2.VideoCapture(1)
cap.set(3, 1280)  # Width of the video frame
cap.set(4, 720)   # Height of the video frame

# Load YOLO model
model = YOLO("C:\\Users\\tians\\Downloads\\Project\\test\\test\\Detect (1).pt")

# Object classes
classNames = ["Female", "Male"]

# Define a flag for stopping the loop
stop_flag = False
frame_queue = Queue()

# Load counts from file or initialize with zeros
if os.path.exists(COUNTS_FILE):
    with open(COUNTS_FILE, "rb") as f:
        counts = pickle.load(f)
else:
    counts = {"Female": 0, "Male": 0}
count_stack = []

# Bounding box overlap threshold
IOU_THRESHOLD = 0.5

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_t, y1_t, x2_t, y2_t = box2

    xA = max(x1, x1_t)
    yA = max(y1, y1_t)
    xB = min(x2, x2_t)
    yB = min(y2, y2_t)

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (x2 - x1 + 1) * (y2 - y1 + 1)
    boxBArea = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def update_image():
    try:
        img = frame_queue.get_nowait()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        label.config(image=img)
        label.image = img
    except Empty:
        pass

    # Update count display
    count_text = f"Female: {counts['Female']}   Male: {counts['Male']}"
    count_label.config(text=count_text)

    root.after(10, update_image)  # Schedule the function to be called again after 10ms

def plot_counts():
    # Extract data from count_stack
    female_counts = [frame['Female'] for frame in count_stack]
    male_counts = [frame['Male'] for frame in count_stack]

    plt.figure(figsize=(10, 5))
    plt.plot(female_counts, label='Female Count')
    plt.plot(male_counts, label='Male Count')
    plt.xlabel('Frame')
    plt.ylabel('Count')
    plt.title('Count of Male and Female Over Time')
    plt.legend()
    plt.show()

def reset_counts():
    global counts, count_stack
    counts = {"Female": 0, "Male": 0}
    count_stack = []
    
    # Save the reset state to file
    with open(COUNTS_FILE, "wb") as f:
        pickle.dump(counts, f)
    
    # Update the count display
    count_text = f"Female: {counts['Female']}   Male: {counts['Male']}"
    count_label.config(text=count_text)

def on_closing():
    global stop_flag
    stop_flag = True
    root.destroy()

def on_key_press(event):
    if event.char == 'r':
        reset_counts()
    elif event.char == 'g':
        plot_counts()
    elif event.char == 'f':  # Toggle fullscreen mode with 'f' key
        toggle_fullscreen()

def toggle_fullscreen():
    global fullscreen
    fullscreen = not fullscreen
    root.attributes('-fullscreen', fullscreen)
    if not fullscreen:
        root.attributes('-topmost', True)
        root.attributes('-topmost', False)

# Create a tkinter window
root = tk.Tk()
root.title("Webcam Feed")

# Initialize fullscreen state
fullscreen = True
root.attributes('-fullscreen', fullscreen)

# Set a minimum size for the root window
root.minsize(1920, 1080)  # Set your desired minimum size

# Set background color and configure the main window
root.configure(bg='#d0f0c0')  # Light green background color

# Create a frame for video display
video_frame = tk.Frame(root, bg='#d0f0c0')
video_frame.pack(fill=tk.BOTH, expand=True)

# Create a label for the video feed
label = tk.Label(video_frame, bg='#d0f0c0')
label.pack(fill=tk.BOTH, expand=True)

# Create a frame for controls
control_frame = tk.Frame(root, bg='#d0f0c0')
control_frame.pack(pady=10)

# Create labels for counts
count_label = tk.Label(control_frame, font=("Helvetica", 24), bg='#d0f0c0', fg='black')  # Increased font size
count_label.pack(pady=10)

# Create buttons with styles
plot_button = tk.Button(control_frame, text="Plot Counts", command=plot_counts, font=("Helvetica", 16), bg='#87ceeb', fg='black', padx=15, pady=10)  # Increased font size and padding
plot_button.pack(side=tk.LEFT, padx=10)

reset_button = tk.Button(control_frame, text="Reset Counts", command=reset_counts, font=("Helvetica", 16), bg='#ff6347', fg='white', padx=15, pady=10)  # Increased font size and padding
reset_button.pack(side=tk.LEFT, padx=10)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.bind('<KeyPress>', on_key_press)  # Bind key press events

# Start the processing thread
thread = threading.Thread(target=process_frame)
thread.start()

# Start the tkinter main loop
root.after(10, update_image)  # Schedule the update_image function to be called after 10ms
root.mainloop()
