from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import json
import threading
import time

app = Flask(__name__)

# Global variables
model = YOLO("yolov8m.pt")
cap = None
detection_count = 0
latest_detections = []
frame_lock = threading.Lock()
current_frame = None

def initialize_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False
    return True

def generate_frames():
    global detection_count, latest_detections, current_frame
    
    while True:
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue
            
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO detection
        results = model.predict(frame, conf=0.8, verbose=False)
        
        detection_count = 0
        current_detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                detection_count += len(boxes)
                
                # Get detection info
                coordinates = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                class_names = result.names
                
                # Draw bounding boxes and collect detection info
                for j in range(len(coordinates)):
                    x1, y1, x2, y2 = coordinates[j]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    score = confidences[j]
                    class_id = classes[j]
                    class_name = class_names[int(class_id)]
                    
                    # Draw rectangle and label
                    color_box = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)
                    cv2.putText(frame, f"{class_name} {score:.2f}", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)
                    
                    # Store detection info
                    current_detections.append({
                        'class': class_name,
                        'confidence': float(score),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        # Update global variables thread-safely
        with frame_lock:
            latest_detections = current_detections
            current_frame = frame.copy()
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_info')
def detection_info():
    with frame_lock:
        return jsonify({
            'count': detection_count,
            'detections': latest_detections
        })

@app.route('/start_camera')
def start_camera():
    if initialize_camera():
        return jsonify({'status': 'success', 'message': 'Camera started'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start camera'})

@app.route('/stop_camera')
def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

if __name__ == '__main__':
    initialize_camera()
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()