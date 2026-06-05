import cv2
import mediapipe as mp
import numpy as np

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
            
            # --- ROBUST THUMB LOGIC (Front & Back Invariant) ---
            thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
            thumb_joint = np.array([hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y])
            
            # measuring extension relative to the opposite side of the palm
            if np.linalg.norm(thumb_tip - pinky_base) > np.linalg.norm(thumb_joint - pinky_base) * 1.05:
                fingers_open.append(1)
            else:
                fingers_open.append(0)
                
            # --- ROBUST 4 FINGERS LOGIC (Front & Back Invariant) ---
            tips = [8, 12, 16, 20]
            knuckles = [6, 10, 14, 18]
            
            for tip_idx, knuckle_idx in zip(tips, knuckles):
                tip = np.array([hand_landmarks.landmark[tip_idx].x, hand_landmarks.landmark[tip_idx].y])
                knuckle = np.array([hand_landmarks.landmark[knuckle_idx].x, hand_landmarks.landmark[knuckle_idx].y])
                
                if np.linalg.norm(tip - wrist) > np.linalg.norm(knuckle - wrist):
                    fingers_open.append(1)
                else:
                    fingers_open.append(0)
                    
            total_fingers = sum(fingers_open)
            print(f"Fingers Up: {total_fingers}")
            
            cv2.putText(frame, f"Fingers: {total_fingers}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            
    cv2.imshow("Stage 10 - Robust Finger Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()