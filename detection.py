import cv2
from ultralytics import YOLO
import tkinter as tk
import numpy as np
from PIL import Image , ImageTk

model = YOLO("yolov8m.pt")

window = tk.Tk ()
window.title("Real - Time Image Display")

# Label para exibir a iamgem
label = tk.Label(window)
label.pack()

text_label = tk.Label(window , text ="")
text_label.pack()

# Open a connection to the webcam (usually device 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    results = model.predict(frame, conf=0.8)

    number_boxes=0
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        number_boxes=number_boxes+len(boxes)
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        #result.show()  # display to screen

        if boxes is not None :
            # Accessar as coordenadas no formato xyxy (x1 , y1 , x2, y2)
            coordinates = boxes.xyxy.cpu().numpy() 
            print(" Bounding box coordinates (xyxy):", coordinates)
            
            confidences = boxes.conf.cpu().numpy()
            print (" Confidence scores :", confidences)

            classes = boxes.cls.cpu().numpy()
            print (" Class predictions :", classes)

            coordinates_xywh = boxes.xywh.cpu().numpy()
            print (" Bounding box coordinates (xywh):", coordinates_xywh)

            class_names = result.names

            for j in range (len(coordinates)):
                x1 , y1 , x2 , y2 = coordinates[j]
                x1 , y1 , x2 , y2 = int(x1), int(y1), int(x2), int(y2)
                
                score = confidences[j]
                class_id = classes[j]
                class_name = class_names[int(class_id)]

                color_box = (0 , 255 , 0)
                cross_size = 7
                cv2.rectangle (frame, ( x1 , y1 ), ( x2 , y2 ), color_box, 2)
                cv2.putText (frame, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

    text_label.config(text = "Number of detections: " + str(number_boxes))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = ImageTk.PhotoImage(img)
    label.configure(image = img)

    label.image = img
    window.update_idletasks()
    window.update()

window.mainloop()