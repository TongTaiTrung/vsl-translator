import sys,os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import config as cfg
from vsl_translator.landmarks import (create_model, detect, draw_landmarks_new, normalize_frame)

def inference(imgArr, holistic):
    if len(imgArr) > 12:
        print("Image array is too big (max 12)")
        return
    for frame in imgArr:
        frame, _meta = normalize_frame(frame)

        img, results = detect(frame, holistic)
        
        
    