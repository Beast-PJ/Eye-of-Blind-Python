import cv2
import numpy as np
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 125)
engine.setProperty('volume', 1.0)

# Initialize YOLO
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

layer_names = yolo.getLayerNames()
output_layers = yolo.getUnconnectedOutLayersNames()

# Open video capture device
vid = cv2.VideoCapture(0)

# Initialize dictionary to keep track of detected objects
detected_objects = {}

while True:
    ret, image = vid.read()
    cv2.imshow('image', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    cv2.imwrite('route.jpg', image)
    
    name = "route.jpg"
    img = cv2.imread(name)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            print("There is ", label)

            if label not in detected_objects:
                # New object detected, say it
                detected_objects[label] = (x, y, w, h)
                if label in ["car", "bicycle", "motorbike", "bus", "chair", "bed", "sofa", "table", "refrigerator"]:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    engine.say("There is " + label + ", take a left or right.")
                    engine.runAndWait()
                elif label == "person":
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    engine.say("There is " + label + ".")
                    engine.runAndWait()
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    engine.say("There is " + label + ".")
                    engine.runAndWait()
                cv2.putText(img, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            else:
                # Object already detected, check if its position changed
                prev_x, prev_y, prev_w, prev_h = detected_objects[label]
                if abs(x - prev_x) > 50 or abs(y - prev_y) > 50:
                    # Position changed, update position and say it
                    detected_objects[label] = (x, y, w, h)
                    if label in ["car", "bicycle", "motorbike", "bus", "chair", "bed", "sofa", "table", "refrigerator"]:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        engine.say("There is " + label + ", take a left or right.")
                        engine.runAndWait()
                    elif label == "person":
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        engine.say("There is " + label + ".")
                        engine.runAndWait()
                    else:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        engine.say("There is " + label + ".")
                        engine.runAndWait()
                    cv2.putText(img, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("image.jpg", img)
    cv2.imwrite("output.jpg", img)

# Release video capture device
vid.release()
cv2.destroyAllWindows()
