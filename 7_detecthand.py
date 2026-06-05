import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # anchor point for all distance calculations
            wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
            pinky_base = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y])
            
            fingers_open = []
            
            # --- THUMB CHECK (distance to pinky base) ---
            thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
            thumb_joint = np.array([hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y])
            
            if np.linalg.norm(thumb_tip - pinky_base) > np.linalg.norm(thumb_joint - pinky_base):
                fingers_open.append(1)
            else:
                fingers_open.append(0)
                
            # --- 4 FINGERS CHECK (distance to wrist) ---
            tips = [8, 12, 16, 20]
            knuckles = [6, 10, 14, 18]
            
            for tip_idx, knuckle_idx in zip(tips, knuckles):
                tip = np.array([hand_landmarks.landmark[tip_idx].x, hand_landmarks.landmark[tip_idx].y])
                knuckle = np.array([hand_landmarks.landmark[knuckle_idx].x, hand_landmarks.landmark[knuckle_idx].y])
                
                # if tip is further from the wrist than the knuckle, it's extended
                if np.linalg.norm(tip - wrist) > np.linalg.norm(knuckle - wrist):
                    fingers_open.append(1)
                else:
                    fingers_open.append(0)
            
            total_fingers = sum(fingers_open)
            
            # simple classification based on total extended count
            if total_fingers == 0:
                status = "FIST"
            elif total_fingers == 5:
                status = "PALM"
            else:
                status = "Moving Hand"
                
            print(f"Detected: {status}")
            cv2.putText(frame, status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
    cv2.imshow("Stage 9 - Invariant Fist & Palm", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()