from ultralytics import YOLO
import cv2

# Загружаем модель
model = YOLO("yolov8n.pt")

# Делаем предсказание на нашей доске
results = model("board.png")

# Выводим, что она нашла
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        print(f"Найдено: {class_name} (уверенность: {confidence:.2f})")