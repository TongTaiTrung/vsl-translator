import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import cv2
import numpy as np

import config as cfg
from vsl_translator.landmarks import create_model, detect, draw_landmarks_test, draw_arm_lengths

# MediaPipe indices: pose L/R shoulder = 11/12, right-hand index fingertip = 8
POSE_L_SHOULDER = 11
POSE_R_SHOULDER = 12
RH_INDEX_TIP = 8


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return
    with create_model() as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image, results = detect(frame, holistic)
            draw_arm_lengths(image, results)
            draw_landmarks_test(image,results)
            h, w = image.shape[:2]

            pose = results.pose_landmarks
            rh = results.right_hand_landmarks
            if pose is not None and rh is not None:
                ls = pose.landmark[POSE_L_SHOULDER]
                rs = pose.landmark[POSE_R_SHOULDER]
                tip = rh.landmark[RH_INDEX_TIP]

                mid = ((ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0)
                dx, dy = tip.x - mid[0], tip.y - mid[1]
                dist = float(np.hypot(dx, dy))
            
                p_tip = (int(tip.x * w), int(tip.y * h))
                p_mid = (int(mid[0] * w), int(mid[1] * h))
                cv2.line(image, p_tip, p_mid, (0, 255, 255), 2)
                cv2.circle(image, p_tip, 5, (0, 0, 255), -1)
                cv2.circle(image, p_mid, 5, (255, 0, 0), -1)
                cv2.putText(image, f"({tip.x:.1f}, {tip.y:.1f})",
                            (p_tip[0] + 10, p_tip[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2, cv2.LINE_AA)
                
                lines = [
                    f"dist: {dist:.1f}",
                    f"dx: {dx:+.1f}  dy: {dy:+.1f}",
                    f"DX: {abs(tip.x-dx):+.1f}  dy: {abs(tip.y-dy):+.1f}",
                    f"tip: ({tip.x:.1f}, {tip.y:.1f})",
                    f"mid: ({mid[0]:.1f}, {mid[1]:.1f})",
                ]
                for i, t in enumerate(lines):
                    cv2.putText(image, t, (20, 40 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2, cv2.LINE_AA)

                print(f"dist={dist:.1f} dx={dx:+.1f} dy={dy:+.1f} "
                      f"tip=({tip.x:.1f},{tip.y:.1f}) mid=({mid[0]:.1f},{mid[1]:.1f})")
            else:
                cv2.putText(image, "missing: no pose/right hand", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2, cv2.LINE_AA)
                print("missing: "
                      f"pose={'ok' if pose else 'no'} "
                      f"right_hand={'ok' if rh else 'no'}")

            cv2.imshow("window", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
