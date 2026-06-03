import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
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
            # tracking mouth corners and center points
            p_left = face_landmarks.landmark[61]    # left corner
            p_right = face_landmarks.landmark[291]  # right corner
            p_top_lip = face_landmarks.landmark[0]   # top of upper lip
            p_left_eye = face_landmarks.landmark[33]
            p_right_eye = face_landmarks.landmark[263]
            
            # 1. measure horizontal stretch relative to eye distance
            mouth_width = np.linalg.norm(np.array([p_left.x, p_left.y]) - np.array([p_right.x, p_right.y]))
            eye_span = np.linalg.norm(np.array([p_left_eye.x, p_left_eye.y]) - np.array([p_right_eye.x, p_right_eye.y]))
            stretch_ratio = mouth_width / eye_span
            
            # 2. measure upward lift (how high corners are relative to the upper lip)
            # in image coordinates, lower Y means higher on screen
            avg_corner_y = (p_left.y + p_right.y) / 2
            upward_lift = p_top_lip.y - avg_corner_y 
            
            # hybrid score combining stretch and upward curl
            smile_score = stretch_ratio + (upward_lift * 2)
            
            # displaying live score for custom calibration
            cv2.putText(frame, f"Smile Score: {round(smile_score, 2)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # trigger smile if it passes calibrated threshold (usually around 0.75 - 0.80)
            if smile_score > 0.78:
                print("Smiling")
                cv2.putText(frame, "SMILING :)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
    cv2.imshow("Stage 4 - Dynamic Smile Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()