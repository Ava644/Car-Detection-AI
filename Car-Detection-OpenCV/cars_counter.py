import argparse
import os
import sys

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Count vehicles in a video using frame differencing.'
    )
    parser.add_argument(
        '-i', '--input', default='cars.mp4',
        help='Path to the input video file or camera index (default: cars.mp4).'
    )
    parser.add_argument(
        '--output', default=None,
        help='Optional output video file path to save the annotated result.'
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='Do not open a display window even if one is available.'
    )
    return parser.parse_args()


def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    return x + x1, y + y1


def create_video_writer(frame, output_path):
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))


def main():
    args = parse_args()

    source = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source '{args.input}'", file=sys.stderr)
        sys.exit(1)

    min_contour_width = 40
    min_contour_height = 40
    offset = 10
    line_height = 550
    matches = []
    cars = 0

    if args.output:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Error: unable to read the first video frame.', file=sys.stderr)
            cap.release()
            sys.exit(1)
        writer = create_video_writer(frame, args.output)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        writer = None

    show_window = not args.no_display and bool(os.environ.get('DISPLAY'))

    ret, frame1 = cap.read()
    if not ret or frame1 is None:
        print('Error: no frames available from the input source.', file=sys.stderr)
        cap.release()
        sys.exit(1)

    ret, frame2 = cap.read()
    if not ret or frame2 is None:
        print('Error: unable to read the second video frame.', file=sys.stderr)
        cap.release()
        sys.exit(1)

    while ret and frame1 is not None and frame2 is not None:
        d = cv2.absdiff(frame1, frame2)
        grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3), np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < min_contour_width or h < min_contour_height:
                continue

            cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)
            cv2.line(frame1, (0, line_height), (1200, line_height), (0, 255, 0), 2)
            centroid = get_centroid(x, y, w, h)
            matches.append(centroid)
            cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

        for mx, my in matches[:]:
            if (line_height - offset) < my < (line_height + offset):
                cars += 1
                matches.remove((mx, my))
                print(cars)

        cv2.putText(
            frame1,
            f"Total Vehicles Detected: {cars}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 170, 0),
            2,
        )

        if writer is not None:
            writer.write(frame1)

        if show_window:
            cv2.imshow('OUTPUT', frame1)
            if cv2.waitKey(1) == 27:
                break

        frame1 = frame2
        ret, frame2 = cap.read()

    cap.release()
    if writer is not None:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
