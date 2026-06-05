import cv2
import mediapipe as mp
import numpy as np
import math

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
            pinky_base = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y])
            
            fingers_open = []
            
            # thumb calculation
            if np.linalg.norm(np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y]) - pinky_base) > np.linalg.norm(np.array([hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y]) - pinky_base) * 1.05:
                fingers_open.append(1)
            else:
                fingers_open.append(0)
                
            # fingers calculation
            for t, k in zip([8, 12, 16, 20], [6, 10, 14, 18]):
                if np.linalg.norm(np.array([hand_landmarks.landmark[t].x, hand_landmarks.landmark[t].y]) - wrist) > np.linalg.norm(np.array([hand_landmarks.landmark[k].x, hand_landmarks.landmark[k].y]) - wrist):
                    fingers_open.append(1)
                else:
                    fingers_open.append(0)
                    
            total_fingers = sum(fingers_open)
            
            # capturing palm location for geometry rendering
            cx = int(hand_landmarks.landmark[9].x * w)
            cy = int(hand_landmarks.landmark[9].y * h)
            shape_size = 60
            
            # --- GEOMETRY ROUTINES ---
            if total_fingers == 1:
                cv2.line(frame, (cx - shape_size, cy), (cx + shape_size, cy), (0, 255, 255), 4)
            elif total_fingers == 2:
                cv2.circle(frame, (cx, cy), shape_size, (0, 255, 0), 3)
            elif total_fingers == 3:
                pts = np.array([[cx, cy - shape_size], [cx - shape_size, cy + shape_size], [cx + shape_size, cy + shape_size]], np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
            elif total_fingers == 4:
                cv2.rectangle(frame, (cx - shape_size, cy - shape_size), (cx + shape_size, cy + shape_size), (0, 165, 255), 3)
            elif total_fingers == 5:
                star_pts = []
                for i in range(10):
                    r = shape_size if i % 2 == 0 else shape_size // 2
                    angle = i * math.pi / 5 - math.pi / 2
                    star_pts.append([int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))])
                cv2.polylines(frame, [np.array(star_pts, np.int32)], isClosed=True, color=(0, 0, 255), thickness=3)
                
    cv2.imshow("Stage 11 - Single Hand Shape Canvas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()