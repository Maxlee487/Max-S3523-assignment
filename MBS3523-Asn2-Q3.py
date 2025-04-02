import cv2
import numpy as np

confThreshold = 0.8
nmsThreshold = 0.4

fruit_prices = {
    'apple': 10,
    'orange': 15,
    'banana': 5
}

cap = cv2.VideoCapture(0)

classesFile = 'C:/Users/ngait\PycharmProjects\PythonProject\coco80.names'  # Make sure this file exists in your directory
classes = []
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNetFromDarknet('C:/Users/ngait\PycharmProjects\PythonProject\yolov3-320.cfg','C:/Users/ngait\PycharmProjects\PythonProject\yolov3-320.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layerNames = net.getLayerNames()
output_layers_names = net.getUnconnectedOutLayersNames()

while True:
    success, img = cap.read()
    if not success:
        break

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    layerOutputs = net.forward(output_layers_names)

    bboxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confThreshold and classes[class_id] in fruit_prices:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bboxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, nmsThreshold)

    font = cv2.FONT_HERSHEY_SIMPLEX

    fruit_counts = {}
    total_price = 0

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = bboxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            if label in fruit_prices:
                if label in fruit_counts:
                    fruit_counts[label] += 1
                else:
                    fruit_counts[label] = 1

                total_price += fruit_prices[label]

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                text = f"{label}: {confidence:.2f}"
                cv2.putText(img, text, (x, y + 20), font, 0.7, (0, 0, 0), 2)
                cv2.putText(img, text, (x, y + 20), font, 0.7, (255, 255, 255), 1)

    total_fruits = sum(fruit_counts.values())
    price_info = f"Total Fruits: {total_fruits} | Price: ${total_price:.2f}"
    cv2.putText(img, price_info, (width - 400, 30), font, 0.7, (0, 0, 0), 2)
    cv2.putText(img, price_info, (width - 400, 30), font, 0.7, (255, 255, 255), 1)

    y_pos = 60
    for fruit, count in fruit_counts.items():
        fruit_info = f"{fruit}: {count}"
        cv2.putText(img, fruit_info, (width - 400, y_pos), font, 0.7, (0, 0, 0), 2)
        cv2.putText(img, fruit_info, (width - 400, y_pos), font, 0.7, (255, 255, 255), 1)
        y_pos += 30

    cv2.imshow('Fruit Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
