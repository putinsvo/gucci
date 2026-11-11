---
language: en
license: mit
tags:
  - yolov8
  - object-detection
  - chess
  - computer-vision
  - ultralytics
model_name: "Dynamic Chess Piece Detector (YOLOv8)"
datasets:
  - custom-chess-dataset
base_model: ultralytics/yolov8
pipeline_tag: object-detection
---

# Dynamic Chess Piece Detection Model (YOLOv8)

This repository provides a Hugging Face–hosted version of a chess piece detection model originally created and trained by **siromermer** in the GitHub project:  
**Dynamic-Chess-Board-Piece-Extraction** (GitHub link: https://github.com/siromermer/Dynamic-Chess-Board-Piece-Extraction).

The intention of this upload is to make the model easier to load in cloud environments such as Google Colab, especially for academic projects where instructors need to pull the weights directly without manual file handling. Full credit for the dataset preparation, initial training, and project concept belongs to the original author.

---

## 📌 Model Overview
This model is based on **YOLOv8** (Ultralytics), a state-of-the-art object detection architecture known for its efficiency and accuracy. The pretrained weights detect individual chess pieces on a board image.

The model was trained on labeled chessboard images using the class set defined by the original author.

---

## 🧩 Class Labels
The model predicts **12 chess piece categories**, separated by color and type.

**Class Names:**
```
['black-bishop', 'black-king', 'black-knight', 'black-pawn', 'black-queen', 'black-rook',
 'white-bishop', 'white-king', 'white-knight', 'white-pawn', 'white-queen', 'white-rook']
```

**Class Dictionary:**
```
{
 0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn',
 4: 'black-queen', 5: 'black-rook', 6: 'white-bishop', 7: 'white-king',
 8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'
}
```

---

## 🔧 Usage Example (Google Colab / Python)
```python
from ultralytics import YOLO

# Load the model directly from Hugging Face
model = YOLO("https://huggingface.co/KanisornPutta/chess-model-yolov8m/resolve/main/chess-model-yolov8m.pt")

# Load a chessboard image
warped_image_path = "path/to/warped_board.jpg"
# Run inference
results = model(warped_image_path)
```

---

## 📝 Attribution
This work is **not an original model** but a re-hosting of pretrained weights created by:
- **Original Author:** siromermer
- **Original Repository:** Dynamic-Chess-Board-Piece-Extraction

This Hugging Face version exists solely for academic and educational convenience, with full acknowledgment of the model's creator.

---

## 📖 License
Follow the license terms defined in the original repository. If none were specified, please contact the original author before using the model commercially.

---

## 📚 References
- Ultralytics YOLOv8 Documentation: https://docs.ultralytics.com
- Dynamic-Chess-Board-Piece-Extraction: https://github.com/siromermer/Dynamic-Chess-Board-Piece-Extraction

