import yaml, time
import cv2, mediapipe as mp
import numpy as np
from detector import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio

# Load config.yaml
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

mp_face = mp.solutions.face_mesh

def get_landmarks(results, w, h):
    if not results.multi_face_landmarks:
        return None
    return [(lm.x * w, lm.y * h) for lm in results.multi_face_landmarks[0].landmark]

def draw_text_box(frame, text, pos, color=(255,255,255), bg_alpha=0.5, scale=0.8):
    """Draw text with semi-transparent background"""
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x-10, y-th-10), (x+tw+10, y+10), (0,0,0), -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, color, 2, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(cfg.get("camera_index", 0))
    cap.set(cv2.CAP_PROP_FPS, cfg.get("fps_target", 30))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Calibrate: Look at the camera and keep your eyes open normally (Press Q to stop)")
    open_vals = []

    window_name = "Calibrate (Press Q or ESC to stop)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as fm:
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)

            lm = get_landmarks(res, w, h)
            if lm:
                ear_l = eye_aspect_ratio(lm, LEFT_EYE)
                ear_r = eye_aspect_ratio(lm, RIGHT_EYE)
                ear = (ear_l + ear_r) / 2
                open_vals.append(ear)

                draw_text_box(frame, f"EAR {ear:.3f}", (30, 100), (0,255,0), 0.6, 1.0)

            # UI text (English)
            draw_text_box(frame, "DriveSafe Eyes-Calibrate Mode", (30, 40), (255,255,0), 0.6, 0.9)
            draw_text_box(frame, "Look straight ahead and keep your eyes open normally.", (30, h-120), (255,255,255), 0.5, 0.8)
            draw_text_box(frame, "Press 'Q' or 'ESC' to stop calibration.", (30, h-70), (180,180,180), 0.5, 0.8)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            if time.time() - t0 > 20:
                break

    cap.release()
    cv2.destroyAllWindows()

    # Process result
    if len(open_vals) < 30:
        print("Not enough frames captured (should be ≥ 30). Please try again.")
        return

    mean = float(np.mean(open_vals))
    std = float(np.std(open_vals))
    thr = max(0.15, min(0.30, mean - 2.0 * std))

    print(f"EAR_open mean={mean:.3f}, std={std:.3f} -> threshold={thr:.3f}")
    cfg["ear_closed_threshold"] = thr
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    print("config.yaml updated successfully ✅")

if __name__ == "__main__":
    main()
