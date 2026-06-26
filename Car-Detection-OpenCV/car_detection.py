import argparse
import os
import sys

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detect cars in a video using a Haar cascade classifier.'
    )
    parser.add_argument(
        '-i', '--input', default='cars.mp4',
        help='Path to input video file or camera index (default: cars.mp4).'
    )
    parser.add_argument(
        '-c', '--cascade', default='haarcascade_cars.xml',
        help='Path to Haar cascade XML file.'
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


def create_video_writer(frame, output_path):
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))


def main():
    args = parse_args()

    if args.input.isdigit():
        source = int(args.input)
    else:
        source = args.input

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source '{args.input}'", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.cascade):
        print(f"Error: cascade file '{args.cascade}' not found.", file=sys.stderr)
        cap.release()
        sys.exit(1)

    car_cascade = cv2.CascadeClassifier(args.cascade)
    if car_cascade.empty():
        print(f"Error: failed to load cascade classifier from '{args.cascade}'", file=sys.stderr)
        cap.release()
        sys.exit(1)

    out = None
    if args.output:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Error: unable to read the first video frame.', file=sys.stderr)
            cap.release()
            sys.exit(1)
        out = create_video_writer(frame, args.output)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    show_window = not args.no_display and bool(os.environ.get('DISPLAY'))

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 3)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if out is not None:
            out.write(frame)

        if show_window:
            cv2.imshow('Car Detection', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cap.release()
    if out is not None:
        out.release()
    if show_window:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
