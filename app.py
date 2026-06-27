from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="dataset/hafif_model/data.yaml",
    epochs=80,
    imgsz=640,
    batch=8,
    device="cpu",
    name="hafif_arac_modeli"
)