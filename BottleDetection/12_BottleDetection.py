import cv2
from ultralytics import YOLO

# loading the freshly baked model weight file
model = YOLO("BottleDetection/runs/detect/train/weights/best.pt")

# opening up the camera capture stream
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        print("Webcam stream disconnected.")
        break
    
    # mirroring frame for natural alignment
    frame = cv2.flip(frame, 1)
    
    # processing the frame to scan for bottles (50% confidence threshold)cant detect so 15% is used
    results = model(frame, conf=0.023)
    
    # using built-in annotation drawing functions to overlay tracking boxes
    annotated_frame = results[0].plot()
    
    # parsing detected classes to output straight to terminal
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        detected_item = model.names[class_id]
        print(f"Terminal Output: Detected a {detected_item}!")
        
    # showing live camera feed window
    cv2.imshow("Live Bottle Detection", annotated_frame)
    
    # press q to drop out of frame tracking
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()