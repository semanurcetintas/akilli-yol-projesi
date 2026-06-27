from ultralytics import YOLO
import cv2
import os
import glob
from collections import deque



LIGHT_MODEL_PATH = "runs/detect/hafif_arac_modeli/weights/best.pt"


VIDEO_PATH = r"C:\teknofest\video3"

CONF_THRESHOLD = 0.45


AREA_HISTORY_SIZE = 8


NEAR_AREA_RATIO_THRESHOLD = 0.22


APPROACH_RATIO_THRESHOLD = 1.10



def resolve_video_path(path):
    if os.path.isfile(path):
        return path

    possible_extensions = ["mp4", "avi", "mov", "mkv"]
    for ext in possible_extensions:
        candidate = f"{path}.{ext}"
        if os.path.isfile(candidate):
            return candidate

    matches = glob.glob(path + ".*")
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Video bulunamadı: {path}")


def trigger_qod_api():
    print("QoD API TETIKLENDI: Yuksek kalite video akisi istenecek.")


def run_strong_model_placeholder(frame):
    cv2.putText(
        frame,
        "GUCLU MODEL AKTIF",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )


video_file = resolve_video_path(VIDEO_PATH)

light_model = YOLO(LIGHT_MODEL_PATH)

cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    raise RuntimeError("Video acilamadi.")

area_history = deque(maxlen=AREA_HISTORY_SIZE)
qod_triggered = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_w * frame_h

    results = light_model(frame, conf=CONF_THRESHOLD, verbose=False)

    best_box = None
    best_conf = 0

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])

            if conf > best_conf:
                best_conf = conf
                best_box = box

    status_text = "ARAC YOK"
    area_ratio = 0

    if best_box is not None:
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()

        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        area_ratio = box_area / frame_area

        area_history.append(area_ratio)

        # Kutu çizimi
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"arac conf: {best_conf:.2f}",
            (int(x1), max(30, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        if len(area_history) >= AREA_HISTORY_SIZE:
            first_area = area_history[0]
            last_area = area_history[-1]

            if first_area > 0 and last_area / first_area > APPROACH_RATIO_THRESHOLD:
                status_text = "ARAC YAKLASIYOR"
            elif first_area > 0 and last_area < first_area:
                status_text = "ARAC UZAKLASIYOR"
            else:
                status_text = "ARAC MESAFE SABIT"
        else:
            status_text = "ARAC TAKIP EDILIYOR"


        if area_ratio >= NEAR_AREA_RATIO_THRESHOLD:
            status_text = "KRITIK MESAFE - QOD / GUCLU MODEL"

            if not qod_triggered:
                trigger_qod_api()
                qod_triggered = True

            run_strong_model_placeholder(frame)


    cv2.rectangle(frame, (10, 10), (650, 125), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Durum: {status_text}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Alan Orani: {area_ratio:.3f} | Esik: {NEAR_AREA_RATIO_THRESHOLD}",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Cikis: Q",
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Hafif Model - Yaklasma Testi", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()