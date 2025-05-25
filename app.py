from flask import Flask, render_template, Response, request, jsonify
import os
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/output_results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始模型與鎖
model_type = 'detect'
model = YOLO('yolo11s.pt')
model_lock = threading.Lock()

# --------- Helper Functions ---------
def draw_bounding_boxes(image, results):
    boxes = results[0].boxes
    names = results[0].names or {}  # 安全取得類別名稱

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().item()
        cls_id = int(box.cls[0].cpu().item())

        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{names.get(cls_id, cls_id)} {conf:.2f}"

        # 畫框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 畫標籤背景
        cv2.rectangle(image, (x1, y1 - 25), (x1 + len(label) * 12, y1), (0, 255, 0), -1)
        # 畫文字
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    return image

def get_model_choices():
    return [
        ('detect', 'Object Detection'),
        ('segment', 'Instance Segmentation'),
        ('classify', 'Image Classification'),
        ('pose', 'Pose Estimation')
    ]

# --------- Real-time Video Feed ---------
def gen_frames():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time

        with model_lock:
            current_model = model
            current_task = model_type

        results = current_model(frame, task=current_task)

        if current_task == 'detect':
            frame = draw_bounding_boxes(frame, results)
        elif current_task in ['segment', 'pose']:
            frame = results[0].plot()
        elif current_task == 'classify':
            names = results[0].names
            probs = getattr(results[0], 'probs', None)
            if probs is not None:
                probs = probs.data.cpu().numpy()
                y = 30
                for i, prob in enumerate(probs):
                    if prob > 0.01:
                        label = f"{names[i]}: {prob:.2f}"
                        cv2.putText(frame, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        y += 30
            else:
                cv2.putText(frame, "No classification results", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 顯示 FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# --------- Routes ---------
@app.route('/')
def index():
    return render_template('index.html', models=get_model_choices(), current_model=model_type)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch_model', methods=['POST'])
def switch_model():
    global model, model_type
    new_type = request.form.get('model_type')
    with model_lock:
        model_type = new_type
        if new_type == 'detect':
            model = YOLO('yolo11s.pt')
        elif new_type == 'segment':
            model = YOLO('yolo11s-seg.pt')
        elif new_type == 'classify':
            model = YOLO('yolo11s-cls.pt')
        elif new_type == 'pose':
            model = YOLO('yolo11s-pose.pt')
    return jsonify({'message': f'Model switched to {new_type}'})

@app.route('/process_image', methods=['POST'])
def process_image():
    image = request.files['image']
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    with model_lock:
        current_model = model
        current_task = model_type
    results = current_model(img, task=current_task)

    if current_task == 'detect':
        img = draw_bounding_boxes(img, results)
    elif current_task in ['segment', 'pose']:
        img = results[0].plot()
    elif current_task == 'classify':
        names = results[0].names
        probs = getattr(results[0], 'probs', None)
        if probs is not None:
            probs = probs.data.cpu().numpy()
            y = 30
            for i, prob in enumerate(probs):
                if prob > 0.01:
                    label = f"{names[i]}: {prob:.2f}"
                    cv2.putText(img, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    y += 30

    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.jpg')
    cv2.imwrite(result_path, img)
    return jsonify(result=result_path)

@app.route('/saved_results')
def saved_results():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify({'results': files})

if __name__ == '__main__':
    app.run(debug=True)
