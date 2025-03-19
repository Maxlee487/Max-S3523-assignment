import serial
import cv2
import threading

SERIAL_PORT = "COM10"
BAUD_RATE = 115200

esp32 = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

light_level = 0

def read_esp32():
    global light_level
    while True:
        try:
            data = esp32.readline().decode().strip()
            if data.isdigit():
                light_level = int(data)
        except Exception as e:
            print(f"Error reading ESP32 data: {e}")

thread = threading.Thread(target=read_esp32)
thread.daemon = True
thread.start()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取相机画面")
        break

    text = f"Light Level: {light_level}"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera with Light Level", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
esp32.close()
