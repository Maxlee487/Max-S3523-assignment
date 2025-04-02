import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def is_finger_extended(landmarks, finger_tip_id, finger_pip_id):
    tip_y = landmarks[finger_tip_id].y
    pip_y = landmarks[finger_pip_id].y

    return tip_y < pip_y


cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_FPS, 45)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

hand_status = "Unknown"

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cam.isOpened():
        ret, img = cam.read()
        if not ret:
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False

        results = hands.process(imgRGB)

        imgRGB.flags.writeable = True
        img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                index_extended = is_finger_extended(hand_landmarks.landmark, 8, 6)
                middle_extended = is_finger_extended(hand_landmarks.landmark, 12, 10)
                ring_extended = is_finger_extended(hand_landmarks.landmark, 16, 14)
                pinky_extended = is_finger_extended(hand_landmarks.landmark, 20, 18)

                if (index_extended and middle_extended and ring_extended) or \
                        (index_extended and middle_extended and pinky_extended) or \
                        (index_extended and ring_extended and pinky_extended) or \
                        (middle_extended and ring_extended and pinky_extended):
                    hand_status = "Open Light"
                else:
                    hand_status = "Close Light"

                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
        else:
            hand_status = "No Hand Detected"

        cv2.rectangle(img, (0, 0), (375, 100), (240, 100, 80), -1)

        cv2.putText(img, 'HAND STATUS', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, hand_status, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Hand Detection', img)

        key = cv2.waitKey(5) & 0xFF
        if key == 32:
            hand_status = "Unknown"
        elif key == 27:  
            break

cam.release()
cv2.destroyAllWindows()
