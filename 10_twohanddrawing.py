import cv2
import mediapipe as mp
import numpy as np
import math

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    tracked_hands = []
    
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
                    
            cx = int(hand_landmarks.landmark[9].x * w)
            cy = int(hand_landmarks.landmark[9].y * h)
            
            tracked_hands.append({'center': (cx, cy), 'count': sum(fingers_open)})

    # --- TWO HAND DISTANCE SCALING CANVAS ---
    if len(tracked_hands) > 0:
        active_shape = tracked_hands[0]['count']
        canvas_center = tracked_hands[0]['center']
        dynamic_size = 60
        
        # if a second hand enters the frame, run scaling calculations
        if len(tracked_hands) == 2:
            h1 = tracked_hands[0]['center']
            h2 = tracked_hands[1]['center']
            
            # center shape precisely between both hands
            canvas_center = ((h1[0] + h2[0]) // 2, (h1[1] + h2[1]) // 2)
            
            # compute linear distance to control shape size
            pixel_dist = math.hypot(h1[0] - h2[0], h1[1] - h2[1])
            dynamic_size = max(15, int(pixel_dist // 2))
            
        cx, cy = canvas_center
        
        if active_shape == 1:
            if len(tracked_hands) == 2:
                cv2.line(frame, tracked_hands[0]['center'], tracked_hands[1]['center'], (0, 255, 255), 5)
            else:
                cv2.line(frame, (cx - dynamic_size, cy), (cx + dynamic_size, cy), (0, 255, 255), 5)
        elif active_shape == 2:
            cv2.circle(frame, (cx, cy), dynamic_size, (0, 255, 0), 3)
        elif active_shape == 3:
            pts = np.array([[cx, cy - dynamic_size], [cx - dynamic_size, cy + dynamic_size], [cx + dynamic_size, cy + dynamic_size]], np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
        elif active_shape == 4:
            cv2.rectangle(frame, (cx - dynamic_size, cy - dynamic_size), (cx + dynamic_size, cy + dynamic_size), (0, 165, 255), 3)
        elif active_shape == 5:
            star_pts = []
            for i in range(10):
                r = dynamic_size if i % 2 == 0 else dynamic_size // 2
                angle = i * math.pi / 5 - math.pi / 2
                star_pts.append([int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))])
            cv2.polylines(frame, [np.array(star_pts, np.int32)], isClosed=True, color=(0, 0, 255), thickness=3)

    cv2.imshow("Stage 12 - Two Hand Scaling Canvas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()