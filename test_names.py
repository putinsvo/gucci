from ultralytics import YOLO

model = YOLO("weights/chess-model-yolov8m.pt")
results = model("board.png")
print("Классы, которые знает модель:", model.names)