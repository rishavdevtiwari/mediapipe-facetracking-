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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # tracking lips inner boundaries
            p_top_lip = face_landmarks.landmark[13]
            p_bottom_lip = face_landmarks.landmark[14]
            p_left_corner = face_landmarks.landmark[61]
            p_right_corner = face_landmarks.landmark[291]
            
            # calculating mouth open dimensions
            mouth_height = np.linalg.norm(np.array([p_top_lip.x, p_top_lip.y]) - np.array([p_bottom_lip.x, p_bottom_lip.y]))
            mouth_width = np.linalg.norm(np.array([p_left_corner.x, p_left_corner.y]) - np.array([p_right_corner.x, p_right_corner.y]))
            
            mouth_ratio = mouth_height / mouth_width
            
            # check if vertical opening passes threshold
            if mouth_ratio > 0.35:
                print("Mouth Open")
                cv2.putText(frame, "MOUTH OPEN", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                
    cv2.imshow("Stage 3 - Mouth Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()