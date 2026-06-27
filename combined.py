from ultralytics import YOLO
import cv2
import os
import glob
import re
from rapidocr_onnxruntime import RapidOCR
import numpy as np
from collections import deque
from typing import Optional


LIGHT_MODEL_PATH = "runs/detect/hafif_arac_modeli/weights/best.pt"
STRONG_MODEL_PATH = "runs/detect/runs/detect/guclu_model_yeni/weights/best.pt"

VIDEO_PATH = r"C:\teknofest\video1"

CONF_LIGHT = 0.45
CONF_STRONG = 0.35

AREA_HISTORY_SIZE = 8
QOD_PRE_TRIGGER = 0.13
STRONG_TRIGGER = 0.25
APPROACH_RATIO = 1.08

PLATE_CLASS_NAME = "plaka"

FLOW_SCALE_MPP = 0.12
AREA_SPEED_K = 95.0

PERSPECTIVE_BOOST_BASE = 0.035
BOOST_LIMIT = 3.0

ocr_model = RapidOCR()

light_model = YOLO(LIGHT_MODEL_PATH)
strong_model = YOLO(STRONG_MODEL_PATH)

LOCKED_PLATE = None
PEAK_AREA = 0.0
PEAK_COOLDOWN = 5
peak_cooldown_counter = 0
was_approaching = False

qod_message = "-"
best_detection_scores = {}

TURKISH_PLATE_PATTERN = re.compile(r'^(\d{2})([A-Z]{1,3})(\d{2,4})$')

OCR_FIX_MAP = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1',
    'Z': '2', 'S': '5',
    'G': '6', 'B': '8',
}


class OpticalFlowSpeedEstimator:
    FEATURE_PARAMS = dict(maxCorners=250, qualityLevel=0.01, minDistance=7, blockSize=7)

    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(self, fps, scale_mpp=0.12, speed_min=1.0, speed_max=180.0, buffer_size=7):
        self.fps = fps
        self.scale_mpp = scale_mpp
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.speed_buffer = deque(maxlen=buffer_size)

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None
        self.speed_buffer.clear()

    def update(self, roi):
        if roi is None or roi.size == 0:
            self.reset()
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is not None and gray.shape != self.prev_gray.shape:
            self._reset_tracking(gray)
            return None

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if self.prev_gray is None:
            self._reset_tracking(gray)
            return None

        if self.prev_pts is None or len(self.prev_pts) < 8:
            self._reset_tracking(gray)
            return None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.LK_PARAMS
        )

        if curr_pts is None or status is None:
            self._reset_tracking(gray)
            return None

        good_prev = self.prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_prev) < 8:
            self._reset_tracking(gray)
            return None

        displacements = np.linalg.norm(good_curr - good_prev, axis=1)

        if len(displacements) == 0:
            self._reset_tracking(gray)
            return None

        q1 = np.percentile(displacements, 25)
        q3 = np.percentile(displacements, 75)
        iqr = q3 - q1

        low = max(0, q1 - 1.5 * iqr)
        high = q3 + 1.5 * iqr

        filtered = displacements[(displacements >= low) & (displacements <= high)]

        if len(filtered) < 5:
            self._update_tracking(gray, good_curr)
            return None

        disp_px = float(np.median(filtered))
        speed_kmh = disp_px * self.scale_mpp * self.fps * 3.6

        if not (self.speed_min <= speed_kmh <= self.speed_max):
            self._update_tracking(gray, good_curr)
            return None

        self.speed_buffer.append(speed_kmh)
        smoothed_speed = round(float(np.median(self.speed_buffer)), 1)

        self._update_tracking(gray, good_curr)
        return smoothed_speed

    def _reset_tracking(self, gray):
        self.prev_gray = gray
        self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.FEATURE_PARAMS)

    def _update_tracking(self, gray, good_curr):
        self.prev_gray = gray

        if good_curr is not None and len(good_curr) > 20:
            self.prev_pts = good_curr.reshape(-1, 1, 2)
        else:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.FEATURE_PARAMS)


class AreaBasedSpeedEstimator:
    def __init__(
        self,
        fps,
        k=95.0,
        boost_base=0.035,
        boost_limit=3.0,
        buffer_size=9,
        speed_min=1.0,
        speed_max=180.0,
    ):
        self.fps = fps
        self.k = k
        self.boost_base = boost_base
        self.boost_limit = boost_limit
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.prev_sqrt_area = None
        self.speed_buffer = deque(maxlen=buffer_size)

    def reset(self):
        self.prev_sqrt_area = None
        self.speed_buffer.clear()

    def update(self, area_ratio):
        if area_ratio <= 0:
            self.reset()
            return None

        sqrt_area = float(np.sqrt(area_ratio))

        if self.prev_sqrt_area is None:
            self.prev_sqrt_area = sqrt_area
            return None

        growth_rate = (sqrt_area - self.prev_sqrt_area) * self.fps
        self.prev_sqrt_area = sqrt_area

        if growth_rate <= 0:
            return None

        perspective_boost = 1.0 + (self.boost_base / max(area_ratio, 0.01))
        perspective_boost = min(perspective_boost, self.boost_limit)

        speed_kmh = growth_rate * self.k * perspective_boost

        if not (self.speed_min <= speed_kmh <= self.speed_max):
            return None

        self.speed_buffer.append(speed_kmh)
        return round(float(np.median(self.speed_buffer)), 1)


def resolve_video_path(path):
    if os.path.isfile(path):
        return path

    for ext in ["mp4", "avi", "mov", "mkv"]:
        candidate = f"{path}.{ext}"
        if os.path.isfile(candidate):
            return candidate

    matches = glob.glob(path + ".*")
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Video bulunamadi: {path}")


def trigger_qod_api():
    global qod_message
    qod_message = "QoD tetiklendi"
    print("QoD API TETIKLENDI")


def clean_plate_text(text):
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9]', '', text)

    if text.startswith("TR") and len(text) > 4:
        text = text[2:]

    if len(text) < 5 or len(text) > 9:
        return ""

    prefix = text[:2]
    fixed_prefix = "".join([OCR_FIX_MAP.get(ch, ch) for ch in prefix])

    rest = text[2:]
    match_digits = re.search(r'\d+$', rest)

    if match_digits:
        suffix = match_digits.group()
        fixed_suffix = "".join([OCR_FIX_MAP.get(ch, ch) for ch in suffix])
        rest = rest[:match_digits.start()] + fixed_suffix
    else:
        suffix = rest[-2:]
        fixed_suffix = "".join([OCR_FIX_MAP.get(ch, ch) for ch in suffix])
        rest = rest[:-2] + fixed_suffix

    return fixed_prefix + rest


def preprocess_plate(plate_crop):
    plate_crop = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    kernel_sharpening = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])

    return cv2.filter2D(gray_clahe, -1, kernel_sharpening)


def read_plate_once(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    plate_w = x2 - x1
    plate_h = y2 - y1

    if plate_w < 20 or plate_h < 8:
        return ""

    pad_x = int(plate_w * 0.06)
    pad_y = int(plate_h * 0.10)

    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(w, int(x2) + pad_x)
    y2 = min(h, int(y2) + pad_y)

    plate_crop = frame[y1:y2, x1:x2]

    if plate_crop.size == 0:
        return ""

    processed_img = preprocess_plate(plate_crop)
    input_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

    result, _ = ocr_model(input_img)

    if not result:
        return ""

    combined_raw_text = ""

    for line in result:
        text = line[1]
        conf = float(line[2])

        if conf < 0.20:
            continue

        combined_raw_text += text

    cleaned = clean_plate_text(combined_raw_text)

    print(f"[BIRLESTIRILMIS HAM METIN]: {combined_raw_text}")
    print(f"[FILTREDEN GECEN SON PLAKA]: {cleaned}")

    if TURKISH_PLATE_PATTERN.match(cleaned):
        return cleaned

    return ""


def combine_speeds(area_speed, flow_speed):
    if area_speed is not None and flow_speed is not None:
        return round((0.70 * area_speed) + (0.30 * flow_speed), 1)

    if area_speed is not None:
        return area_speed

    if flow_speed is not None:
        return flow_speed

    return None


def format_detection_message(label):
    label_lower = label.lower()

    if label_lower == "telefon":
        return "Telefonla konusuyor", "risk"

    if label_lower == "sigara":
        return "Sigara iciyor", "risk"

    if label_lower == "saga_bakis":
        return "Surucu saga bakiyor", "risk"

    if label_lower == "sola_bakis":
        return "Surucu sola bakiyor", "risk"

    if label_lower == "nesne":
        return "Nesne tespit edildi", "object"

    return None, None


def draw_modern_panel(frame, plate_text, speed_text, qod_text, risk_text, object_text, status_text):
    overlay = frame.copy()

    x, y = 20, 20
    w, h = 560, 250

    cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 210, 255), 2)

    cv2.putText(frame, "AKILLI YOL GUVENLIGI SISTEMI",
                (x + 20, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 255), 2)

    cv2.line(frame, (x + 20, y + 50), (x + w - 20, y + 50), (90, 90, 90), 1)

    cv2.putText(frame, f"Durum: {status_text}",
                (x + 20, y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    cv2.putText(frame, f"Plaka: {plate_text}",
                (x + 20, y + 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.putText(frame, f"Tahmini Hiz: {speed_text}",
                (x + 20, y + 144),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    qod_color = (0, 255, 0) if qod_text != "-" else (170, 170, 170)

    cv2.putText(frame, f"QoD: {qod_text}",
                (x + 20, y + 176),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, qod_color, 2)

    risk_color = (0, 0, 255) if risk_text != "-" else (170, 170, 170)

    cv2.putText(frame, f"Riskli Durum: {risk_text}",
                (x + 20, y + 208),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, risk_color, 2)

    cv2.putText(frame, f"Nesne Tespiti: {object_text}",
                (x + 20, y + 238),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 180, 0), 2)


def draw_confidence_table(frame, detection_table):
    overlay = frame.copy()

    x, y = 600, 20
    w, h = 390, 250

    cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 210, 255), 2)

    cv2.putText(frame, "TESPIT GUVEN TABLOSU",
                (x + 20, y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2)

    cv2.line(frame, (x + 20, y + 50), (x + w - 20, y + 50), (90, 90, 90), 1)

    cv2.putText(frame, "Sinif", (x + 20, y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 2)

    cv2.putText(frame, "Guven", (x + 280, y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 2)

    if not detection_table:
        cv2.putText(frame, "Tespit yok",
                    (x + 20, y + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        return

    row_y = y + 110

    for item in detection_table[:5]:
        label = item["label"]
        conf_percent = item["conf"] * 100

        if item["type"] == "risk":
            color = (0, 0, 255)
        elif item["type"] == "object":
            color = (255, 180, 0)
        else:
            color = (180, 180, 180)

        cv2.putText(frame, label,
                    (x + 20, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        cv2.putText(frame, f"%{conf_percent:.1f}",
                    (x + 280, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)

        row_y += 30


video_file = resolve_video_path(VIDEO_PATH)
cap = cv2.VideoCapture(video_file)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

flow_speed_estimator = OpticalFlowSpeedEstimator(
    fps=fps,
    scale_mpp=FLOW_SCALE_MPP,
    speed_min=1.0,
    speed_max=180.0,
    buffer_size=7
)

area_speed_estimator = AreaBasedSpeedEstimator(
    fps=fps,
    k=AREA_SPEED_K,
    boost_base=PERSPECTIVE_BOOST_BASE,
    boost_limit=BOOST_LIMIT,
    buffer_size=9,
    speed_min=1.0,
    speed_max=180.0
)

area_history = deque(maxlen=AREA_HISTORY_SIZE)

qod_triggered = False
strong_active = False

current_speed = None
area_speed_value = None
flow_speed_value = None

speed_lost_counter = 0
SPEED_LOST_LIMIT = 12

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w = frame.shape[:2]
    frame_area = h * w

    detection_table = []

    light_results = light_model(frame, conf=CONF_LIGHT, verbose=False)

    best_box = None
    best_conf = 0.0

    for result in light_results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box

    status = "ARAC YOK"
    area_ratio = 0.0
    approaching = False

    if best_box is not None:
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()

        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / frame_area
        area_history.append(area_ratio)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        cv2.putText(frame, f"arac {best_conf:.2f}",
                    (int(x1), max(25, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        new_area_speed = area_speed_estimator.update(area_ratio)

        if new_area_speed is not None:
            area_speed_value = new_area_speed

        vx1, vy1, vx2, vy2 = map(int, [x1, y1, x2, y2])

        vx1 = max(0, vx1)
        vy1 = max(0, vy1)
        vx2 = min(w, vx2)
        vy2 = min(h, vy2)

        vehicle_roi = frame[vy1:vy2, vx1:vx2]

        if vehicle_roi.size > 0 and (vx2 - vx1) > 50 and (vy2 - vy1) > 50:
            vehicle_roi = cv2.resize(vehicle_roi, (320, 320), interpolation=cv2.INTER_LINEAR)
            new_flow_speed = flow_speed_estimator.update(vehicle_roi)

            if new_flow_speed is not None:
                flow_speed_value = new_flow_speed
        else:
            flow_speed_estimator.reset()

        combined_speed = combine_speeds(area_speed_value, flow_speed_value)

        if combined_speed is not None:
            current_speed = combined_speed
            speed_lost_counter = 0
        else:
            speed_lost_counter += 1

        if speed_lost_counter > SPEED_LOST_LIMIT:
            current_speed = None
            area_speed_value = None
            flow_speed_value = None
            area_speed_estimator.reset()
            flow_speed_estimator.reset()

        if len(area_history) == AREA_HISTORY_SIZE:
            first = area_history[0]
            last = area_history[-1]

            if first > 0 and last / first >= APPROACH_RATIO:
                approaching = True
                status = "ARAC YAKLASIYOR"
            elif last < first:
                status = "ARAC UZAKLASIYOR"
            else:
                status = "MESAFE SABIT"
        else:
            status = "TAKIP EDILIYOR"

        if approaching and area_ratio >= QOD_PRE_TRIGGER and not qod_triggered:
            trigger_qod_api()
            qod_triggered = True

        if approaching and area_ratio >= STRONG_TRIGGER:
            strong_active = True

        if was_approaching and not approaching and area_ratio < PEAK_AREA:
            peak_cooldown_counter = PEAK_COOLDOWN

        if area_ratio > PEAK_AREA:
            PEAK_AREA = area_ratio

        was_approaching = approaching

    else:
        speed_lost_counter += 1

        if speed_lost_counter > SPEED_LOST_LIMIT:
            current_speed = None
            area_speed_value = None
            flow_speed_value = None
            area_speed_estimator.reset()
            flow_speed_estimator.reset()

    if strong_active:
        status = "GUCLU MODEL AKTIF"

        strong_results = strong_model(frame, conf=CONF_STRONG, verbose=False)

        for result in strong_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = strong_model.names[cls_id]

                bx1, by1, bx2, by2 = box.xyxy[0].tolist()

                if label == PLATE_CLASS_NAME:
                    box_color = (0, 255, 255)
                elif label in ["telefon", "sigara", "saga_bakis", "sola_bakis"]:
                    box_color = (0, 0, 255)
                else:
                    box_color = (255, 120, 0)

                cv2.rectangle(frame,
                              (int(bx1), int(by1)),
                              (int(bx2), int(by2)),
                              box_color,
                              2)

                cv2.putText(frame,
                            f"{label} {conf:.2f}",
                            (int(bx1), max(25, int(by1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            box_color,
                            2)

                message, message_type = format_detection_message(label)

                if message is not None:

                    key = message

                    if key not in best_detection_scores:
                        best_detection_scores[key] = {
                            "label": message,
                            "type": message_type,
                            "conf": conf
                        }

                    elif conf > best_detection_scores[key]["conf"]:
                        best_detection_scores[key] = {
                            "label": message,
                            "type": message_type,
                            "conf": conf
                        }

                    detection_table = list(best_detection_scores.values())

                if label == PLATE_CLASS_NAME:

                    if LOCKED_PLATE is not None:
                        pass

                    elif peak_cooldown_counter > 0 or (
                        approaching and area_ratio >= STRONG_TRIGGER
                    ):
                        plate_text = read_plate_once(frame, bx1, by1, bx2, by2)

                        if plate_text:
                            LOCKED_PLATE = plate_text
                            print(f"[KILITLENDI] {LOCKED_PLATE}")

                        if peak_cooldown_counter > 0:
                            peak_cooldown_counter -= 1

                    if LOCKED_PLATE:
                        cv2.putText(frame,
                                    f"PLAKA: {LOCKED_PLATE}",
                                    (int(bx1), int(by2) + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 255),
                                    2)

    risk_items = [item for item in detection_table if item["type"] == "risk"]
    object_items = [item for item in detection_table if item["type"] == "object"]

    if risk_items:
        risk_message = " | ".join([item["label"] for item in risk_items])
    else:
        risk_message = "-"

    if object_items:
        object_message = "nesne tespit edildi"
    else:
        object_message = "nesne yok"

    plate_display = LOCKED_PLATE if LOCKED_PLATE else "-"
    speed_display = f"{current_speed:.1f} km/h" if current_speed is not None else "-"

    draw_modern_panel(
        frame,
        plate_text=plate_display,
        speed_text=speed_display,
        qod_text=qod_message,
        risk_text=risk_message,
        object_text=object_message,
        status_text=status
    )

    draw_confidence_table(frame, detection_table)

    cv2.imshow("Akilli Yol Guvenligi Sistemi", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()