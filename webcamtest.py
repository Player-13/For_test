import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    cv2.imwrite('opencv_test.jpg', frame)
    print("Frame captured successfully")
else:
    print("Failed to capture frame")
cap.release()
