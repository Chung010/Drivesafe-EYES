import math, time, collections, csv, os
from typing import Iterable, List, Sequence, Tuple

# ดัชนี landmark รอบตา (MediaPipe FaceMesh)
LEFT_EYE: List[int]  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE: List[int] = [362, 387, 385, 263, 373, 380]

Point = Tuple[float, float]

def dist(a: Point, b: Point) -> float:
    """ระยะยูคลิดระหว่างจุด 2 จุด"""
    return math.dist(a, b)

def eye_aspect_ratio(landmarks: Sequence[Point], eye_idx: Iterable[int]) -> float:
    """
    คำนวณ EAR จากจุดตา 6 จุด: p1..p6
    EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4|| + 1e-6)
    หมายเหตุ: +1e-6 ป้องกันหารศูนย์
    """
    idx = list(eye_idx)
    if len(idx) < 6:  # กันพลาดในเฟรมที่จุดไม่ครบ
        return 0.0
    try:
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idx[:6]]
    except Exception:
        return 0.0
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)

class EARSmoother:
    """ค่า EMA สำหรับทำ smoothing (ใช้กับ EAR/MAR ได้)"""
    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = float(alpha)
        self.val: float | None = None

    def update(self, x: float) -> float:
        self.val = x if self.val is None else (self.alpha * x + (1 - self.alpha) * self.val)
        return self.val

class MajorityVote:
    """โหวตแบบเสียงข้างมากในหน้าต่าง k เฟรม"""
    def __init__(self, k: int = 5) -> None:
        self.k = int(k)
        self.buf: "collections.deque[bool]" = collections.deque(maxlen=self.k)

    def push(self, v: bool) -> None:
        self.buf.append(bool(v))

    def closed(self) -> bool:
        if not self.buf:
            return False
        # ต้อง “มากกว่าครึ่ง” ถึงจะถือว่า True
        return sum(self.buf) > (len(self.buf) // 2)

def face_quality_check(
    frame_gray, face_bbox: Tuple[int, int, int, int] | None,
    min_face_ratio: float = 0.08, min_brightness: float = 40
) -> Tuple[bool, str]:
    """
    ตรวจคุณภาพภาพเบื้องต้น:
    - ความสว่างเฉลี่ยต้อง >= min_brightness
    - กล่องหน้าต้องมีสัดส่วน >= min_face_ratio ของทั้งเฟรม
    """
    h, w = frame_gray.shape[:2]
    brightness = float(frame_gray.mean())
    if brightness < min_brightness:
        return False, f"low light ({brightness:.1f})"

    if face_bbox is None:
        return False, "no face"

    x, y, wf, hf = face_bbox
    face_ratio = (wf * hf) / max(w * h, 1)
    if face_ratio < min_face_ratio:
        return False, f"face too small ({face_ratio:.3f})"

    return True, "ok"

def ensure_logs() -> str:
    """สร้างโฟลเดอร์/ไฟล์ log ถ้ายังไม่มี และคืน path"""
    os.makedirs("logs", exist_ok=True)
    path = "logs/events.csv"
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "EAR", "closed", "alert"])
    return path

def log_event(path: str, ear: float, closed: bool | int, alert: str = "") -> None:
    """เติมบรรทัด log ใหม่ (ปลอดภัยกับเครื่องหมายจุลภาคเพราะใช้ CSV writer)"""
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([time.time(), f"{ear:.4f}", int(bool(closed)), alert])
