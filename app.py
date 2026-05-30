import os, yaml, time, math, sys
import cv2, mediapipe as mp
from detector import (
    LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, EARSmoother,
    MajorityVote, face_quality_check, ensure_logs, log_event
)

DEBUG_PRINT = False

# -------------------- ดัชนีปากจาก MediaPipe FaceMesh (468 จุด) --------------------
# มุมปากซ้าย/ขวา (outer corners): 78, 308
# ริมฝีปากบน/ล่าง (inner mid):    13, 14
MOUTH_LEFT_CORNER  = 78
MOUTH_RIGHT_CORNER = 308
MOUTH_UPPER        = 13
MOUTH_LOWER        = 14

def mouth_aspect_ratio(pts):
    """
    pts: list[(x,y)] ของจุด FaceMesh 468 จุด
    MAR = ระยะบน-ล่าง / ระยะซ้าย-ขวา (เลี่ยงหาร 0)
    """
    lx, ly = pts[MOUTH_LEFT_CORNER]
    rx, ry = pts[MOUTH_RIGHT_CORNER]
    ux, uy = pts[MOUTH_UPPER]
    dx, dy = pts[MOUTH_LOWER]
    horiz = math.hypot(rx - lx, ry - ly)
    vert  = math.hypot(dx - ux, dy - uy)  
    return vert / horiz if horiz > 1e-6 else 0.0


# -------------------- เสียงแจ้งเตือน (พร้อมป้องกัน error) --------------------
sound_alert1 = None
sound_alert2 = None
PYGAME_AVAILABLE = True

try:
    import pygame
except Exception as e:
    PYGAME_AVAILABLE = False
    print(f"[WARNING] ไม่สามารถ import pygame ได้: {e}")
    print("→ โปรแกรมจะทำงานต่อโดย 'ไม่มีเสียงเตือน'")
else:
    try:
        pygame.mixer.init()
        # โหลดไฟล์เสียงแบบกันพัง: ถ้าไฟล์ไม่มี ให้เตือนเฉย ๆ
        if os.path.exists('assets/alert1.wav'):
            sound_alert1 = pygame.mixer.Sound('assets/alert1.wav')
        else:
            print("[WARNING] ไม่พบไฟล์เสียง assets/alert1.wav")

        if os.path.exists('assets/alert2.wav'):
            sound_alert2 = pygame.mixer.Sound('assets/alert2.wav')
        else:
            print("[WARNING] ไม่พบไฟล์เสียง assets/alert2.wav")
    except Exception as e:
        PYGAME_AVAILABLE = False
        sound_alert1 = None
        sound_alert2 = None
        print(f"[WARNING] pygame.mixer ใช้งานไม่ได้: {e}")
        print("→ โปรแกรมจะทำงานต่อโดย 'ไม่มีเสียงเตือน'")

def play_alert1():
    if PYGAME_AVAILABLE and sound_alert1 is not None:
        try:
            pygame.mixer.stop()
            sound_alert1.play()
        except Exception as e:
            print(f"[WARNING] เล่นเสียง alert1 ไม่ได้: {e}")

def play_alert2():
    if PYGAME_AVAILABLE and sound_alert2 is not None:
        try:
            pygame.mixer.stop()
            sound_alert2.play()
        except Exception as e:
            print(f"[WARNING] เล่นเสียง alert2 ไม่ได้: {e}")

# --- เสียงสำหรับ "หาวครบหลายครั้ง" (optional) ---
sound_yawn_series = None
if PYGAME_AVAILABLE:
    try:
        if os.path.exists('assets/yawn_series.wav'):
            sound_yawn_series = pygame.mixer.Sound('assets/yawn_series.wav')
            sound_yawn_series.set_volume(1.0)   # 🔊 ปรับความดังที่นี่
        else:
            # fallback: ใช้ alert2 หากไม่มี yawn_series.wav
            sound_yawn_series = sound_alert2
            if sound_yawn_series is not None:
                sound_yawn_series.set_volume(1.0)
            else:
                print("[WARNING] ไม่พบ assets/yawn_series.wav และไม่มีเสียง fallback")
    except Exception as e:
        print(f"[WARNING] โหลดเสียง yawn_series ไม่ได้: {e}")
        sound_yawn_series = None


def play_yawn_series():
    if PYGAME_AVAILABLE and sound_yawn_series is not None:
        try:
            pygame.mixer.stop()
            sound_yawn_series.play()
        except Exception as e:
            print(f"[WARNING] เล่นเสียง yawn_series ไม่ได้: {e}")

# -------------------- คอนฟิก --------------------
try:
    with open("config.yaml","r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
except FileNotFoundError:
    print("[ERROR] ไม่พบไฟล์ config.yaml ในโฟลเดอร์ปัจจุบัน")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] อ่านไฟล์ config.yaml ไม่ได้: {e}")
    sys.exit(1)
UI = float(cfg.get("ui_scale", 1.0))

# ขนาด/สัดส่วนที่ใช้ซ้ำ
BAR_H      = int(56 * UI)   # ความสูงแถบหัว
TITLE_FS   = 0.8 * UI       # font scale title
SUB_FS     = 0.6 * UI       # font scale subtitle
TITLE_TH   = max(2, int(2 * UI))  # ความหนาเส้นตัวอักษร

CHIP_FS    = 0.6 * UI
CHIP_TH    = max(2, int(2 * UI))
CHIP_PAD   = int(8 * UI)

LINE_STEP  = int(36 * UI)   # ระยะห่างระหว่างบรรทัดชิป
BASE_Y     = int(72 * UI)   # เริ่มวางชิปแถวแรก




# -------------------- ธีม & ยูทิล --------------------
THEME = {
    "OK":   ( 60,180, 75),   # เขียว BGR
    "WARN": (  0,215,255),   # เหลืองทอง
    "ALRM": (  0,  0,255),   # แดง
    "PANEL_BG_ALPHA": 0.55,  # ความทึบของกล่อง
}

def overlay_alpha(img, overlay, x, y, alpha=1.0):
    """วางภาพ overlay (BGRA หรือ BGR) ลงบน img ที่ตำแหน่ง (x,y) พร้อมโปร่งใส"""
    h, w = overlay.shape[:2]
    if overlay.shape[2] == 4:
        alpha_m = overlay[:,:,3:] / 255.0 * alpha
        alpha_b = 1.0 - alpha_m
        y1, y2 = max(y,0), min(y+h, img.shape[0])
        x1, x2 = max(x,0), min(x+w, img.shape[1])
        oy1, oy2 = y1-y, y1-y+(y2-y1)
        ox1, ox2 = x1-x, x1-x+(x2-x1)
        if y1<y2 and x1<x2:
            img[y1:y2, x1:x2, :3] = (alpha_m[oy1:oy2, ox1:ox2]*overlay[oy1:oy2, ox1:ox2, :3] +
                                     alpha_b[oy1:oy2, ox1:ox2]*img[y1:y2, x1:x2, :3])
    else:
        roi = img[y:y+h, x:x+w]
        cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, roi)

def draw_title_bar(frame, title, state, logo):
    h, w = frame.shape[:2]
    color = THEME[state]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, BAR_H), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (0,0), (int(12*UI), BAR_H), color, -1)

    x = int(20*UI)
    if logo is not None:
        lh = BAR_H - int(16*UI)
        scale = lh / logo.shape[0]
        lw = int(logo.shape[1] * scale)
        logo_rs = cv2.resize(logo, (lw, lh), interpolation=cv2.INTER_AREA)
        overlay_alpha(frame, logo_rs, x, int(8*UI), 1.0)
        x += lw + int(12*UI)

    # Title
    cv2.putText(frame, title, (x, int(23*UI)),
                cv2.FONT_HERSHEY_SIMPLEX, TITLE_FS, (0,0,0), TITLE_TH+1, cv2.LINE_AA)
    cv2.putText(frame, title, (x, int(23*UI)),
                cv2.FONT_HERSHEY_SIMPLEX, TITLE_FS, (255,255,255), TITLE_TH, cv2.LINE_AA)

    # Subtitle
    sub = {"OK":"Normal","WARN":"Keep Eyes Open","ALRM":"Eyes Closed!"}[state]
    cv2.putText(frame, sub, (x, int(46*UI)),
                cv2.FONT_HERSHEY_SIMPLEX, SUB_FS, (0,0,0), TITLE_TH+1, cv2.LINE_AA)
    cv2.putText(frame, sub, (x, int(46*UI)),
                cv2.FONT_HERSHEY_SIMPLEX, SUB_FS, color, TITLE_TH, cv2.LINE_AA)


def draw_chip(frame, label, value, x, y, color=(255,255,255)):
    s = f"{label}: {value}"
    (tw, th), bl = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, CHIP_FS, CHIP_TH)
    pad = CHIP_PAD
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - th - pad), (x + tw + pad*2, y + bl + pad), (0,0,0), -1)
    cv2.addWeighted(overlay, THEME["PANEL_BG_ALPHA"], frame, 1 - THEME["PANEL_BG_ALPHA"], 0, frame)
    cv2.putText(frame, s, (x + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, CHIP_FS, (0,0,0), CHIP_TH+1, cv2.LINE_AA)
    cv2.putText(frame, s, (x + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, CHIP_FS, color, CHIP_TH, cv2.LINE_AA)


def load_logo(path):
    if not os.path.exists(path): return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # รองรับ PNG โปร่งใส
    return img

def _open_camera_with_fallback(index:int, fps_target:int):
    """
    เปิดกล้องด้วย fallback:
      1) CAP_DSHOW (Windows บางเครื่องลื่นกว่า)
      2) ค่า default
    คืนค่า: cv2.VideoCapture ที่เปิดสำเร็จแล้ว
    """
    cap = None
    # ลองด้วย CAP_DSHOW ก่อน (จะไม่มีผลบนระบบที่ไม่รองรับ)
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FPS, fps_target)
            return cap
        if cap is not None:
            cap.release()
    except Exception:
        pass  # ตกไปลองวิธีถัดไป

    # ลองแบบปกติ
    cap = cv2.VideoCapture(index)
    if cap is not None and cap.isOpened():
        cap.set(cv2.CAP_PROP_FPS, fps_target)
        return cap

    # ถ้าไม่สำเร็จเลย ให้ raise error
    raise RuntimeError(f"Cannot open camera at index {index}")

def main():
    global DEBUG_PRINT

    # ---------- ค่าคงที่/คอนฟิก ----------
    thr_closed      = float(cfg.get("ear_closed_threshold", 0.25))
    closed_min      = float(cfg.get("closed_min_seconds", 1.2))               # Alert1
    alert2_more_sec = float(cfg.get("alert2_after_more_seconds", 2.0))        # เพิ่มอีกจน Alert2
    alpha           = float(cfg.get("smoothing_alpha", 0.4))
    kvote           = int(cfg.get("majority_frames", 3))
    min_face_ratio  = float(cfg.get("min_face_ratio", 0.05))
    min_brightness  = float(cfg.get("min_brightness", 30))
    allow_low_qual  = bool(cfg.get("allow_alert_on_low_quality", True))

    # MAR / Yawning
    mar_thr         = float(cfg.get("mar_threshold", 0.60))
    yawn_min        = float(cfg.get("yawn_min_seconds", 1.0))
    mar_alpha       = float(cfg.get("mar_smoothing_alpha", 0.4))
    yawn_kvote      = int(cfg.get("yawn_majority_frames", 3))

    # --- Yawn series (หาวหลายครั้งภายในหน้าต่างเวลา) ---
    yawn_series_needed     = int(cfg.get("yawn_series_needed", 3))        # ต้องหาวกี่ครั้ง
    yawn_series_window_sec = float(cfg.get("yawn_series_window_sec", 60)) # ภายในกี่วินาที

    # ป้ายแจ้งเตือนหาว (sticky)
    yawn_notice_min_show = float(cfg.get("yawn_notice_min_show_sec", 2.0))
    yawn_notice_cooldown = float(cfg.get("yawn_notice_cooldown_sec", 5.0))

    # ---------- อุปกรณ์/ตัวช่วย ----------
    camera_index = int(cfg.get("camera_index", 0))
    fps_target   = int(cfg.get("fps_target", 30))

    try:
        cap = _open_camera_with_fallback(camera_index, fps_target)
    except Exception as e:
        print(f"[ERROR] กล้องไม่สามารถเปิดได้ (index={camera_index}): {e}")
        print("→ ตรวจสอบว่า: (1) กล้องเชื่อมต่ออยู่ (2) ค่า camera_index ใน config.yaml ถูกต้อง (3) แอปอื่นไม่ได้ใช้กล้องอยู่")
        sys.exit(1)

    #  ตั้งความละเอียดหลังเปิดกล้อง 'สำเร็จ' (720p; ถ้าจะลอง 1080p ให้เปลี่ยนเป็น 1920x1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # logs
    try:
        log_path = ensure_logs()
    except Exception as e:
        print(f"[WARNING] ไม่สามารถสร้างโฟลเดอร์/ไฟล์ logs ได้: {e}")
        print("→ ระบบจะยังรันต่อไป แต่จะไม่บันทึกเหตุการณ์")
        log_path = None

    smooth     = EARSmoother(alpha=alpha)
    vote       = MajorityVote(k=kvote)
    mar_smooth = EARSmoother(alpha=mar_alpha)
    yawn_vote  = MajorityVote(k=yawn_kvote)

    last_closed_start = None
    alert1_fired_this_close = False
    alert2_fired_this_close = False

    yawn_start = None
    yawn_fired = False
    yawn_notice_until = 0.0          # เวลา (epoch) ที่ป้ายควรแสดงค้างถึง
    last_yawn_notice_at = 0.0        # เวลาแสดงป้ายครั้งล่าสุด (สำหรับ cooldown)

    # --- สถานะนับ "หาวหลายครั้ง" ---
    yawn_series_count = 0
    yawn_series_first_ts = None

    session_start = time.time()
    logo = load_logo("assets/logo.png")

    try:
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                             max_num_faces=1,
                                             refine_landmarks=True,
                                             min_detection_confidence=0.5,
                                             min_tracking_confidence=0.5) as fm:
            
            # ตั้งให้เต็มจอเมื่อเปิดโปรแกรม
            cv2.namedWindow("DriveSafe Eyes", cv2.WINDOW_NORMAL) #ถ้าอยากให้กลับไปไม่เต็มจอลบ  2 บรรทัดนี้ออก
            cv2.setWindowProperty("DriveSafe Eyes", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  #ถ้าอยากให้กลับไปไม่เต็มจอลบ  2 บรรทัดนี้ออก

            while True:
                try:
                    ret, frame = cap.read()
                except Exception as e:
                    print(f"[ERROR] อ่านภาพจากกล้องล้มเหลว: {e}")
                    break

                if not ret or frame is None:
                    print("[ERROR] ไม่สามารถอ่านภาพจากกล้องได้ (ret=False)")
                    break

                frame = cv2.flip(frame, 1)
                
                h, w  = frame.shape[:2]

                try:    
                    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                except Exception as e:
                    print(f"[ERROR] แปลงสีภาพไม่ได้: {e}")
                    break

                try:
                    res   = fm.process(rgb)
                except Exception as e:
                    print(f"[ERROR] ประมวลผล FaceMesh ผิดพลาด: {e}")
                    break

                alert = ""
                ok, reason = False, "no face"
                closed_voted = False
                ear_s = 0.0

                # เตรียมสถานะ Yawn สำหรับกรณีไม่มีหน้า
                is_yawning = False
                yawn_alert_text = ""
                yawn_dur_disp = 0.0

                if res.multi_face_landmarks:
                    xs, ys = [], []
                    for p in res.multi_face_landmarks[0].landmark:
                        xs.append(int(p.x*w)); ys.append(int(p.y*h))
                    x1, x2 = max(min(xs),0), min(max(xs), w-1)
                    y1, y2 = max(min(ys),0), min(max(ys), h-1)
                    bbox   = (x1, y1, x2-x1, y2-y1)

                    # ---- EAR / ตาปิด ----
                    pts   = [(p.x*w, p.y*h) for p in res.multi_face_landmarks[0].landmark]
                    ear   = (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0
                    ear_s = smooth.update(ear)

                    is_closed = ear_s < thr_closed
                    vote.push(is_closed)
                    closed_voted = vote.closed()

                    ok, reason = face_quality_check(gray, bbox, min_face_ratio, min_brightness)
                    effective_ok = ok or allow_low_qual

                    # ---- MAR / Yawning ----
                    mar_raw = mouth_aspect_ratio(pts)
                    mar_s   = mar_smooth.update(mar_raw)
                    is_yawn_frame = mar_s > mar_thr
                    yawn_vote.push(is_yawn_frame)
                    is_yawning = yawn_vote.closed()

                    # จัดการทริกเกอร์/ค้างป้ายหาว
                    if effective_ok:
                        now = time.time()
                        if is_yawning:
                            if yawn_start is None:
                                yawn_start = now
                                yawn_fired = False
                            yawn_dur = now - yawn_start
                            if (yawn_dur >= yawn_min) and not yawn_fired \
                               and (now - last_yawn_notice_at >= yawn_notice_cooldown):
                                # ทริกเกอร์หาว: ตั้งให้ป้ายแสดงค้าง
                                yawn_alert_text = "Yawning Detected"
                                yawn_fired = True
                                last_yawn_notice_at = now
                                yawn_notice_until = now + yawn_notice_min_show

                                # --- นับจำนวน "เหตุการณ์หาว" ภายในหน้าต่างเวลา ---
                                if (yawn_series_first_ts is None) or ((now - yawn_series_first_ts) > yawn_series_window_sec):
                                    # เริ่มหน้าต่างเวลาใหม่
                                    yawn_series_first_ts = now
                                    yawn_series_count = 0

                                yawn_series_count += 1

                                # ถ้าหาวครบตามกำหนดภายในหน้าต่างเวลา -> เล่นเสียงพิเศษ แล้วรีเซ็ตหน้าต่าง
                                if yawn_series_count >= yawn_series_needed:
                                    play_yawn_series()
                                    # เพิ่มแท็กลง log ของรอบนี้ (จะถูกเขียนในบล็อก log ด้านล่าง)
                                    yawn_alert_text += "|YAWN_SERIES"
                                    # รีเซ็ตหน้าต่างหลังแจ้งเตือน
                                    yawn_series_first_ts = None
                                    yawn_series_count = 0
                        else:
                            yawn_start = None
                            yawn_fired = False
                    else:
                        yawn_start = None
                        yawn_fired = False

                    # ---- แจ้งเตือนตาปิด ----
                    if effective_ok:
                        now = time.time()
                        if closed_voted:
                            if last_closed_start is None:
                                last_closed_start = now
                                alert1_fired_this_close = False
                                alert2_fired_this_close = False
                            dur = now - last_closed_start
                            if (dur >= closed_min) and not alert1_fired_this_close:
                                play_alert1(); alert1_fired_this_close = True; alert = "ALERT1"
                            if (dur >= closed_min + alert2_more_sec) and not alert2_fired_this_close:
                                play_alert2(); alert2_fired_this_close = True; alert = "ALERT2"
                        else:
                            last_closed_start = None
                            alert1_fired_this_close = False
                            alert2_fired_this_close = False
                    else:
                        alert = f"QUALITY:{reason}"

                    # ---- บันทึก log (รวม YAWN ถ้ามี) ----
                    log_alert = alert
                    if yawn_alert_text:
                        # ถ้ามี YAWN_SERIES ติดท้ายจากด้านบน จะใส่รวมเข้ามาด้วย
                        log_alert = (log_alert + "|" + yawn_alert_text) if log_alert else yawn_alert_text
                    if 'log_path' in locals() and log_path:
                        try:
                            log_event(log_path, ear_s, int(closed_voted), log_alert)
                        except Exception as e:
                            if DEBUG_PRINT:
                                print(f"[WARNING] เขียน log ไม่ได้: {e}")

                    # ---- ตัวเลขอินดิเคเตอร์ ----
                    elapsed  = int(time.time() - session_start)
                    hh = elapsed // 3600; mm = (elapsed % 3600)//60; ss = elapsed % 60
                    elapsed_str = f"{hh:02d}:{mm:02d}:{ss:02d}"
                    yawn_dur_disp = (time.time() - yawn_start) if (is_yawning and yawn_start) else 0.0
                    closed_dur = (time.time() - last_closed_start) if (closed_voted and last_closed_start) else 0.0

                    # ---- สีสถานะ (รวมอาการหาวเป็น WARN เมื่อกำลังหาวหรือป้ายยังค้าง) ----
                    yawn_notice_active = (time.time() < yawn_notice_until)
                    if closed_voted and alert2_fired_this_close:
                        state = "ALRM"
                    elif closed_voted or is_yawning or yawn_notice_active:
                        state = "WARN"
                    else:
                        state = "OK"

                    # ====== วาด UI ======
                    draw_title_bar(frame, "DriveSafe Eyes", state, logo)
                    color = THEME[state]
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    
                    base_y = BASE_Y
                    draw_chip(frame, "EAR",      f"{ear_s:.3f} (TH {thr_closed:.3f})", int(14*UI), base_y);            base_y += LINE_STEP
                    draw_chip(frame, "Duration", f"{elapsed_str}",                    int(14*UI), base_y);            base_y += LINE_STEP
                    draw_chip(frame, "Closed",   f"{closed_dur:0.1f}s",               int(14*UI), base_y, (0,255,255)); base_y += LINE_STEP
                    
                    base_y += int(LINE_STEP * 0.8)
                     
                    draw_chip(frame, "MAR",      f"{mar_s:.3f} (TH {mar_thr:.2f})",   int(14*UI), base_y);            base_y += LINE_STEP
                    draw_chip(frame, "Yawn",     f"{yawn_dur_disp:0.1f}s",            int(14*UI), base_y, (0,255,255)); base_y += LINE_STEP
                    draw_chip(frame, "YawnCount",f"{yawn_series_count}/{yawn_series_needed}", int(14*UI), base_y);     base_y += LINE_STEP
                    if yawn_notice_active:
                        draw_chip(frame, "Yawning", "Detected", int(14*UI), base_y, (0,0,255))


                    if DEBUG_PRINT:
                        print(f"EAR:{ear_s:.3f} TH:{thr_closed:.3f} | MAR:{mar_s:.3f} TH:{mar_thr:.2f} "
                              f"| closed:{int(ear_s < thr_closed)}/{int(closed_voted)} | yawn:{int(is_yawning)} "
                              f"| ok:{ok} | alert:{log_alert} "
                              f"| yawn_series:{yawn_series_count}/{yawn_series_needed} "
                              f"| yawn_notice_active:{int(yawn_notice_active)}")

                else:
                    # ไม่มีหน้า: รีเซ็ตสถานะ eye/yawn (จะไม่ไปยุ่งกับ yawn_notice_until เพื่อให้ป้ายค้างครบเวลา)
                    yawn_start = None
                    yawn_fired = False
                    last_closed_start = None
                    alert1_fired_this_close = False
                    alert2_fired_this_close = False

                    draw_title_bar(frame, "DriveSafe Eyes", "WARN", logo)
                    cv2.putText(frame, "NO FACE", (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                
                
                # ไว้เช็ค fps ใน debug
                #fps_now = cap.get(cv2.CAP_PROP_FPS)
                #if DEBUG_PRINT:
                 #print(f"FPS now: {fps_now}")
    
                #ไว้ดู fps    
                #fps = cap.get(cv2.CAP_PROP_FPS)
               #cv2.putText(frame, f"{fps:.0f} FPS", (w-100, 30),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
   

                try:
                    cv2.imshow("DriveSafe Eyes", frame)
                    key = cv2.waitKey(1) & 0xFF
                except Exception as e:
                    print(f"[ERROR] แสดงหน้าต่างภาพไม่ได้: {e}")
                    break

                if key == ord('q') or key == 27: break
                if key == ord('d'):
                    DEBUG_PRINT = not DEBUG_PRINT

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    # เวลาเริ่ม session สำหรับตัวจับเวลา
    session_start = time.time()
    main()
