import cv2
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
from deepface import DeepFace
import threading
import base64
import torch

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# 載入 YOLOv5 模型
model = torch.hub.load('yolov5', 'yolov5s', source='local', pretrained=True)
model.conf = 0.5

frame_queue = []
frame_lock = threading.Lock()
current_mode = "object"
last_face_analysis_time = 0  # 控制人臉分析頻率

@app.route('/')
def index():
    return render_template('index.html')

def video_capture():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = camera.read()
        if success:
            with frame_lock:
                if len(frame_queue) > 10:
                    frame_queue.pop(0)
                frame_queue.append(frame)
        time.sleep(1 / 30)

def object_detection():
    while True:
        if current_mode == "object":
            with frame_lock:
                if frame_queue:
                    raw_frame = frame_queue[-1].copy()
                else:
                    time.sleep(1 / 30)
                    continue

            frame = raw_frame.copy()
            results = model(frame)
            detected_objects = []

            for *xyxy, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                label = model.names[int(cls)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                detected_objects.append(label)

            _, raw_buffer = cv2.imencode('.jpg', raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            _, processed_buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            socketio.emit('video_frame', {
                'raw_frame': base64.b64encode(raw_buffer).decode('utf-8'),
                'processed_frame': base64.b64encode(processed_buffer).decode('utf-8'),
            })

            socketio.emit('update_labels', {'labels': detected_objects})
            socketio.emit('play_alert_sound', {'play_sound': 'person' in detected_objects})
        else:
            time.sleep(0.5)
        time.sleep(1 / 30)

def face_detection():
    global last_face_analysis_time
    while True:
        if current_mode == "face":
            now = time.time()
            if now - last_face_analysis_time >= 1:#調整人臉辨識速度
                with frame_lock:
                    if frame_queue:
                        frame = frame_queue[-1].copy()
                    else:
                        time.sleep(1 / 30)
                        continue
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=['gender', 'age', 'emotion'],
                        enforce_detection=False
                    )

                    faces_info = []
                    for face in result:
                        faces_info.append({
                            'gender': face['dominant_gender'],
                            'age': face['age'],
                            'emotion': face['dominant_emotion']
                        })

                    # 傳送分析結果
                    socketio.emit('face_analysis', {'faces': faces_info})

                    # 顯示人臉框 + 顯示影像畫面（處理後）
                    processed_frame = frame.copy()
                    for face in result:
                        region = face['region']
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    _, raw_buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    _, processed_buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

                    socketio.emit('video_frame', {
                        'raw_frame': base64.b64encode(raw_buffer).decode('utf-8'),
                        'processed_frame': base64.b64encode(processed_buffer).decode('utf-8'),
                    })

                    last_face_analysis_time = now
                except Exception as e:
                    print("Face detection error:", e)
        else:
            time.sleep(0.5)
        time.sleep(1 / 30)


@socketio.on('switch_mode')
def handle_switch_mode(data):
    global current_mode
    new_mode = data['mode']
    if new_mode != current_mode:
        print(f"[後端] 模式切換中：{current_mode} ➜ {new_mode}")
        with frame_lock:
            frame_queue.clear()  # 清空 queue 避免切換時用到舊資料
        current_mode = new_mode
        time.sleep(0.3)  # 小延遲避免 race condition
    print(f"[後端] 已切換到 {current_mode} 模式")

if __name__ == '__main__':
    threading.Thread(target=video_capture, daemon=True).start()
    threading.Thread(target=object_detection, daemon=True).start()
    threading.Thread(target=face_detection, daemon=True).start()

    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
