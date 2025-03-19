import cv2
import numpy as np
import serial
import time

try:
    ser = serial.Serial('COM9', baudrate=9600, timeout=1)
    time.sleep(2)
except serial.SerialException as e:
    print("序列埠開啟錯誤:", e)
    exit()

def send_servo_positions(angle_h, angle_v):

    data = f"{angle_h},{angle_v}\r"
    try:
        ser.write(data.encode())
        print(f"送出角度 - 水平: {angle_h} 垂直: {angle_v}")
    except serial.SerialException as e:
        print("傳送失敗:", e)

def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.createTrackbar('HueLow', 'Trackbars', 35, 179, nothing)
cv2.createTrackbar('HueHigh', 'Trackbars', 85, 179, nothing)
cv2.createTrackbar('SatLow', 'Trackbars', 50, 255, nothing)
cv2.createTrackbar('SatHigh', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('ValLow', 'Trackbars', 50, 255, nothing)
cv2.createTrackbar('ValHigh', 'Trackbars', 255, 255, nothing)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

prev_angle_h = 90
prev_angle_v = 90
alpha = 0.09

area_threshold = 2000

dead_zone = 2

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影像")
        break

    hueLow = cv2.getTrackbarPos('HueLow', 'Trackbars')
    hueHigh = cv2.getTrackbarPos('HueHigh', 'Trackbars')
    satLow = cv2.getTrackbarPos('SatLow', 'Trackbars')
    satHigh = cv2.getTrackbarPos('SatHigh', 'Trackbars')
    valLow = cv2.getTrackbarPos('ValLow', 'Trackbars')
    valHigh = cv2.getTrackbarPos('ValHigh', 'Trackbars')

    lower_hsv = np.array([hueLow, satLow, valLow])
    upper_hsv = np.array([hueHigh, satHigh, valHigh])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area > area_threshold:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx = x + w // 2
            cy = y + h // 2

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

            target_angle_h = int(np.interp(cx, [0, frame_width], [180, 0]))
            target_angle_v = int(np.interp(cy, [0, frame_height], [0, 180]))

            if abs(target_angle_h - prev_angle_h) > dead_zone:
                smooth_angle_h = int(alpha * target_angle_h + (1 - alpha) * prev_angle_h)
            else:
                smooth_angle_h = prev_angle_h

            if abs(target_angle_v - prev_angle_v) > dead_zone:
                smooth_angle_v = int(alpha * target_angle_v + (1 - alpha) * prev_angle_v)
            else:
                smooth_angle_v = prev_angle_v

            prev_angle_h = smooth_angle_h
            prev_angle_v = smooth_angle_v

            send_servo_positions(smooth_angle_h, smooth_angle_v)

            cv2.putText(frame, f"H: {smooth_angle_h} V: {smooth_angle_v}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Object too small or lost", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No object detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
