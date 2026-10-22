#!/usr/bin/env python3
"""Open the webcam and display MediaPipe Holistic landmarks in real time."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
from vsl_translator.landmarks import create_model, detect, draw_landmarks, extract
import config as cfg


def main():
    with create_model() as model:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image, results = detect(frame, model)
            frame = draw_landmarks(frame, results)

            print(len(extract(results, frame_shape=frame.shape[:2])))

            cv2.imshow("window", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
