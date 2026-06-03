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
            # tracing eyes using standard mediapipe indices
            # left eye: top, bottom, outer left, inner right landmarks
            p_top = face_landmarks.landmark[159]
            p_bottom = face_landmarks.landmark[145]
            p_left = face_landmarks.landmark[33]
            p_right = face_landmarks.landmark[133]
            
            # calculating eye aspect ratio math
            vertical_dist = np.linalg.norm(np.array([p_top.x, p_top.y]) - np.array([p_bottom.x, p_bottom.y]))
            horizontal_dist = np.linalg.norm(np.array([p_left.x, p_left.y]) - np.array([p_right.x, p_right.y]))
            
            # scale-invariant ratio of eye height to width
            eye_ratio = vertical_dist / horizontal_dist
            
            # evaluating if ratio dropped significantly
            if eye_ratio < 0.15:
                print("Blinked")
                cv2.putText(frame, "BLINKED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
    cv2.imshow("Stage 2 - Blink Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()