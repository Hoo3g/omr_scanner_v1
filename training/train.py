from ultralytics import YOLO

def train_model():
    # Load a model (yolo11n-obb is a good lightweight choice for OBB)
    model = YOLO("yolo11n-obb.pt")

    # Cấu hình huấn luyện "Pro" để đạt độ chính xác cao nhất
    results = model.train(
        data="dataset.yaml",
        epochs=300,
        imgsz=800,           # Độ phân giải cân bằng tốt cho GTX 1660 Ti
        batch=4,
        patience=100,        # Cho phép học lâu hơn để đạt mốc mAP cao nhất
        optimizer='AdamW',   # AdamW thường ổn định và chính xác hơn cho bài toán này
        lr0=0.001,           # Học phí khởi điểm nhỏ cho AdamW
        label_smoothing=0.1, # Giúp mô hình linh hoạt hơn, không bị "học vẹt"
        dropout=0.1,
        cos_lr=True,
        close_mosaic=20,     # TẮT tạo ảnh ghép ở 20 epoch cuối để "siết" viền cho chuẩn
        amp=False,           
        # --- Augmentation (Đã có dữ liệu aug_ nên chỉnh nhẹ lại) ---
        degrees=10.0,
        perspective=0.0005,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        # ------------------------------------------------------------
        device="0",
        project="runs/detect",
        name="answer_sheet_v4_pro"
    )
    print("Training complete! Weights saved in runs/detect/answer_sheet_v3_augmented/weights")

if __name__ == "__main__":
    train_model()
