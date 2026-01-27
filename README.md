# OMR AI Scanning System v1.0

Hệ thống chấm điểm OMR thông minh sử dụng YOLOv11, hỗ trợ quét trực tiếp từ điện thoại và tự động thu thập dữ liệu để cải thiện mô hình.

## Cấu trúc thư mục
- `app.py`: Server Flask (Backend).
- `omr_pipeline_v1.py`: Nhân xử lý OMR (Pipeline).
- `bubble_labeler_v2.py`: Công cụ gắn nhãn và chỉnh sửa lỗi dự đoán.
- `models/`: Chứa các trọng số AI đã huấn luyện (.pt).
- `templates/`: Giao diện Web di động.
- `collected_data/`: Nơi lưu trữ ảnh quét từ điện thoại để tái huấn luyện.

## Hướng dẫn cài đặt
1. Cài đặt Python 3.9+
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

## Hướng dẫn sử dụng
### 1. Chạy Web Scanner (Cho điện thoại)
```bash
python app.py
```
- Truy cập từ máy tính: `http://localhost:5000`
- Truy cập từ điện thoại: `http://<IP_MAY_TINH>:5000` (Dùng chung Wi-Fi).

### 2. Gắn nhãn & Sửa lỗi (Để cải thiện AI)
Khi có ảnh quét từ điện thoại, bạn có thể mở công cụ này để sửa các lỗi dự đoán của AI:
```bash
python bubble_labeler_v2.py
```
- **S**: Lưu và chuyển ảnh tiếp theo.
- **A/D**: Chuyển ảnh.
- **Click chuột**: Đổi trạng thái (Empty/Filled).
- **Shift + Click**: Thêm ô tròn mới.

## 3. Huấn luyện lại mô hình (Training)
Nếu bạn đã thu thập đủ dữ liệu mới trong `collected_data` và muốn nâng cấp AI:

### A. Huấn luyện phát hiện vùng (Region Detection - OBB)
Dùng để nhận diện các khối Answers, SBD, Made trên phiếu:
```bash
cd training
python train.py
```
*Lưu ý: Cần chuẩn bị folder `dataset_ai` (chứa ảnh và nhãn OBB).*

### B. Huấn luyện phát hiện ô tròn (Bubble Detection)
Dùng để nhận diện ô nào được tô, ô nào trống:
1. **Augmentation (Tăng cường dữ liệu)**: Tạo thêm các mẫu ô tô mờ, chói sáng...
   ```bash
   python augment_filled.py
   ```
2. **Train**:
   ```bash
   python train_bubbles.py
   ```
*Lưu ý: Cần chuẩn bị folder `dataset_bubble_ai`.*

## Lưu ý
- **GPU**: Khuyến khích sử dụng máy tính có GPU NVIDIA để quá trình huấn luyện diễn ra nhanh chóng.
- **Trọng số mới**: Sau khi train xong, hãy lấy file `best.pt` trong thư mục `runs/` và ghi đè vào thư mục `models/` của chương trình chính.
