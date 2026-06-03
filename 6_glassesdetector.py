import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

# setting up face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # identifying bounds of the lower face region (where a mask sits)
            nose = face_landmarks.landmark[4]
            chin = face_landmarks.landmark[152]
            left_bounding = face_landmarks.landmark[234]
            right_bounding = face_landmarks.landmark[454]
            
            # converting to actual pixel coordinates
            ny = int(nose.y * h)
            cy = int(chin.y * h)
            lx = int(left_bounding.x * w)
            rx = int(right_bounding.x * w)
            
            # expanding the box slightly down past the chin to get full coverage
            y1 = max(0, ny)
            y2 = min(h, cy + 10)
            x1 = max(0, lx)
            x2 = min(w, rx)
            
            if x2 > x1 and y2 > y1:
                # cropping out the mouth and jaw area
                lower_face_roi = frame[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                
                # converting to HSV to accurately track skin color channels
                hsv_roi = cv2.cvtColor(lower_face_roi, cv2.COLOR_BGR2HSV)
                
                # standard HSV boundaries for human skin tones
                lower_skin = np.array([0, 20, 70], dtype="uint8")
                upper_skin = np.array([20, 255, 255], dtype="uint8")
                
                # isolating pixels that match skin tones
                skin_mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)
                
                # OPTIONAL DEBUG WINDOW: Shows detected skin as white, masks as black
                cv2.imshow("Skin Detection Map", skin_mask)
                
                # calculating what percentage of the lower face is actual skin
                total_pixels = skin_mask.size
                skin_pixels = cv2.countNonZero(skin_mask)
                skin_percentage = (skin_pixels / total_pixels) * 100
                
                # displaying live metrics on the camera frame
                cv2.putText(frame, f"Skin Area: {round(skin_percentage, 1)}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Threshold logic: If the lower face is less than 35% skin, a mask is present
                if skin_percentage < 35:
                    print(f"Mask Status: WEARING MASK ({round(skin_percentage, 1)}% skin)")
                    cv2.putText(frame, "MASK DETECTED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    print(f"Mask Status: NO MASK ({round(skin_percentage, 1)}% skin)")
                    cv2.putText(frame, "NO MASK", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
    cv2.imshow("Stage 8 - Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()