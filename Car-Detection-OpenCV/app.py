import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_SOURCE = 0
VIDEO_SOURCES = {
    'video': os.path.join(BASE_DIR, 'cars.mp4'),
    'parking': os.path.join(BASE_DIR, 'parking_lot.mp4'),
    'traffic': os.path.join(BASE_DIR, 'traffic.mp4'),
}
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_cars.xml')

# Map of parking slots for sources that support slot tracking. Placeholder
# coordinates (x1, y1, x2, y2) for each slot; user will adjust later.
PARKING_SLOTS = {
    'parking': [
        {'id': 1, 'box': (50, 100, 150, 200)},
        {'id': 2, 'box': (160, 100, 260, 200)},
        {'id': 3, 'box': (270, 100, 370, 200)},
        {'id': 4, 'box': (380, 100, 480, 200)},
    ]
}
stats_lock = threading.Lock()
detection_stats = {
    'source': 'video',
    'count': 0,
    'status': 'Waiting',
    'slots': [],
    'total': 0,
    'occupied': 0,
    'vacant': 0,
}


def compute_slot_status(source_key, car_boxes, scale=1.0):
    slots = PARKING_SLOTS.get(source_key)
    if not slots:
        return []

    results = []
    for slot in slots:
        x1, y1, x2, y2 = slot['box']
        if scale != 1.0:
            x1 = int(x1 * scale)
            y1 = int(y1 * scale)
            x2 = int(x2 * scale)
            y2 = int(y2 * scale)
        slot_area = max(0, x2 - x1) * max(0, y2 - y1)
        occupied = False

        for (cx, cy, cw, ch) in car_boxes:
            cx1, cy1, cx2, cy2 = int(cx), int(cy), int(cx + cw), int(cy + ch)
            ix1 = max(x1, cx1)
            iy1 = max(y1, cy1)
            ix2 = min(x2, cx2)
            iy2 = min(y2, cy2)
            if ix2 > ix1 and iy2 > iy1 and slot_area > 0:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if (inter / float(slot_area)) > 0.3:
                    occupied = True
                    break

        results.append({'id': slot['id'], 'status': 'Occupied' if occupied else 'Vacant'})

    return results


def update_stats(source, count, car_boxes, scale=1.0):
    with stats_lock:
        detection_stats['source'] = source
        detection_stats['count'] = count
        detection_stats['status'] = get_status_label(count)

        slots = compute_slot_status(source, car_boxes, scale=scale)
        detection_stats['slots'] = slots
        total = len(slots)
        occupied = sum(1 for s in slots if s.get('status') == 'Occupied')
        vacant = total - occupied
        detection_stats['total'] = total
        detection_stats['occupied'] = occupied
        detection_stats['vacant'] = vacant


def get_stats():
    with stats_lock:
        return dict(detection_stats)


def get_status_label(count):
    if count == 0:
        return 'Empty lot'
    if count <= 5:
        return 'Partially occupied'
    return 'Busy/full'


def get_source_path(source_key):
    if source_key == 'camera':
        return CAMERA_SOURCE
    return VIDEO_SOURCES.get(source_key, VIDEO_SOURCES['video'])


def get_detection_params(source_key):
    if source_key == 'parking':
        return {'scaleFactor': 1.02, 'minNeighbors': 2, 'minSize': (24, 24), 'maxSize': (220, 220)}
    if source_key == 'traffic':
        return {'scaleFactor': 1.05, 'minNeighbors': 4, 'minSize': (35, 35), 'maxSize': (220, 220)}
    return {'scaleFactor': 1.06, 'minNeighbors': 3, 'minSize': (30, 30), 'maxSize': (220, 220)}


def get_stream_settings(source_key):
    if source_key == 'parking':
        return {'process_every': 2, 'max_width': 960, 'jpeg_quality': 60}
    if source_key == 'traffic':
        return {'process_every': 2, 'max_width': 720, 'jpeg_quality': 60}
    return {'process_every': 2, 'max_width': 720, 'jpeg_quality': 65}


def resize_for_stream(frame, source_key):
    height, width = frame.shape[:2]
    settings = get_stream_settings(source_key)
    max_width = settings['max_width']
    scale = min(1.0, max_width / float(width))
    if scale < 1.0:
        resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        resized = frame
    return resized, scale


def generate_frames(source_key):
    source = get_source_path(source_key)
    if isinstance(source, str) and not os.path.exists(source):
        raise RuntimeError(f'Video file not found: {source}')

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video source: {source}')

    car_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if car_cascade.empty():
        raise RuntimeError(f'Cannot load cascade: {CASCADE_PATH}')

    detect_params = get_detection_params(source_key)
    stream_settings = get_stream_settings(source_key)
    process_every = stream_settings['process_every']
    frame_counter = 0
    back_subtractor = None
    if source_key == 'parking':
        back_subtractor = cv2.createBackgroundSubtractorMOG2(history=180, varThreshold=35, detectShadows=False)

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_counter += 1
        if frame_counter % process_every != 0:
            continue

        display_frame, scale = resize_for_stream(frame, source_key)
        if source_key == 'parking' and back_subtractor is not None:
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            fg_mask = back_subtractor.apply(gray)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cars = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                if area < 400 or w < 24 or h < 24:
                    continue
                cars.append((x, y, w, h))
        else:
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            gray = cv2.equalizeHist(gray)
            cars = car_cascade.detectMultiScale(gray, **detect_params)

        count = len(cars)
        update_stats(source_key, count, cars, scale=scale)

        for (x, y, w, h) in cars:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode(
            '.jpg',
            display_frame,
            [cv2.IMWRITE_JPEG_QUALITY, stream_settings['jpeg_quality']],
        )
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

    cap.release()


@app.route('/')
def index():
    return render_template(
        'index.html',
        page_title='Car Detection Research Dashboard | Live Vehicle Analytics'
    )


@app.route('/video_feed')
def video_feed():
    source_key = request.args.get('src', 'video')
    if source_key not in ('video', 'camera', 'parking', 'traffic'):
        source_key = 'video'
    return Response(generate_frames(source_key),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    return jsonify(get_stats())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
