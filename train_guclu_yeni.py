from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="dataset/guclu_model_yeni/data.yaml",
    epochs=100,
    imgsz=768,
    batch=4,
    device=0,
    workers=0,
    project="runs/detect",
    name="guclu_model_yeni",
    exist_ok=False
)