"""Chay thu VSLInference voi model that: 12 frame den -> predict -> in ket qua."""

import cv2
import numpy as np

from VSLInference import VSLInference

MODEL_PATH = "backend/models/model.keras"

# frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(12)]

cap = cv2.VideoCapture(0)
frames = []

while len(frames)<12:
    ret,frame=cap.read()
    if not ret: break
    frames.append(frame)
    cv2.imshow("TEST",frame)
    cv2.waitKey(10)


with VSLInference(MODEL_PATH) as vsl:
    print("labels:", vsl._labels)
    out = vsl.predict(frames)

print("label:", repr(out["label"]))
print("confidence:", out["confidence"])
print("probs:", [round(float(p), 4) for p in out["probs"]])