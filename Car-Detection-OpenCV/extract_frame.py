import cv2
import sys

import os

VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'parking_lot.mp4')
OUT_FILE = 'frame_sample.png'

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f'ERROR: Cannot open video file: {VIDEO_PATH}', file=sys.stderr)
    sys.exit(2)

success, frame = cap.read()
if not success or frame is None:
    print('ERROR: Failed to read first frame', file=sys.stderr)
    cap.release()
    sys.exit(3)

# Save sample frame in current directory
ok = cv2.imwrite(OUT_FILE, frame)
if not ok:
    print(f'ERROR: Failed to write {OUT_FILE}', file=sys.stderr)
    cap.release()
    sys.exit(4)

height, width = frame.shape[:2]
print(f'Width: {width}')
print(f'Height: {height}')

cap.release()
