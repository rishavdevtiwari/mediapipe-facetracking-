import cv2
import mediapipe as mp
import numpy as np

# setting up camera feed
cap = cv2.VideoCapture(0)

# loading the face mesh 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        print("Can't find webcam feed.")
        break
        
    # flipping frame for a natural mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # default status text
    horizontal_status = "Looking Center"
    vertical_status = "Looking Center"
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # key landmarks for tracking head rotation
            nose_tip = face_landmarks.landmark[4]
            left_cheek = face_landmarks.landmark[454]  # screen left (user's left)
            right_cheek = face_landmarks.landmark[234] # screen right (user's right)
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            
            # --- YAW LOGIC (Left / Right) ---
            # measuring horizontal distance from nose to both cheeks
            dist_left = abs(nose_tip.x - left_cheek.x)
            dist_right = abs(nose_tip.x - right_cheek.x)
            
            # preventing division by zero error
            if dist_right == 0: dist_right = 0.001
            yaw_ratio = dist_left / dist_right
            
            # validating thresholds for horizontal turns
            if yaw_ratio < 0.55:
                horizontal_status = "Turned Left"
            elif yaw_ratio > 1.75:
                horizontal_status = "Turned Right"
                
            # --- PITCH LOGIC (Up / Down) ---
            # measuring vertical distance from nose to forehead and chin
            dist_up = abs(nose_tip.y - forehead.y)
            dist_down = abs(chin.y - nose_tip.y)
            
            if dist_down == 0: dist_down = 0.001
            pitch_ratio = dist_up / dist_down
            
            # validating thresholds for vertical tilts
            # (at rest, this ratio naturally sits around 0.9 - 1.1)
            if pitch_ratio < 0.65:
                vertical_status = "Looking Up"
            elif pitch_ratio > 1.45:
                vertical_status = "Looking Down"
                
            # combining outputs if moving diagonally, otherwise keeping it simple
            if horizontal_status != "Looking Center" or vertical_status != "Looking Center":
                output_text = f"{horizontal_status} | {vertical_status}".replace(" | Looking Center", "").replace("Looking Center | ", "")
            else:
                output_text = "Center"
                
            # print output to the terminal window
            print(f"Head Position: {output_text}")
            
            # displaying direction text on the webcam preview
            cv2.putText(frame, output_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            
            # drawing visual anchor points on the face for debugging
            for landmark in [nose_tip, left_cheek, right_cheek, forehead, chin]:
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    cv2.imshow("Stage 6 - Head Pose Detection", frame)
    
    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()