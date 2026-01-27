from ultralytics import YOLO

# 1. Create bubble_data.yaml
with open("bubble_data.yaml", "w") as f:
    f.write(f"""
path: ../dataset_bubble_ai
train: images
val: images

names:
  0: num
  1: empty
  2: filled
""")

# 2. Train the model
def train_bubble_model():
    model = YOLO("yolo11n.pt") # Use regular YOLO11n for bubbles
    model.train(
        data="bubble_data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,           # Giảm batch size để tránh hết bộ nhớ GPU
        project="runs/detect",
        name="bubble_detector_v1"
    )

if __name__ == "__main__":
    train_bubble_model()
