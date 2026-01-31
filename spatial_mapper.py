import numpy as np

def omr_decoder(detections):
    """
    YOLO là 'Đôi mắt': Phát hiện các chấm đen.
    Decoder này là 'Bộ não': Phân tích tọa độ thành Câu hỏi và Đáp án.
    
    detections: list các kết quả YOLO [{ 'box': [cx, cy, w, h], 'class': 1, 'conf': 0.9 }, ...]
    Tọa độ normalized từ 0.0 đến 1.0
    """
    # 1. Chỉ lấy các thực thể được YOLO gán nhãn là 'filled' (ô đã tô)
    # Giả sử class 1 là 'filled'
    filled_bubbles = [d for d in detections if d['class'] == 1]
    
    if not filled_bubbles:
        return {"status": "error", "message": "Không tìm thấy ô tô đen nào."}

    # 2. Gom nhóm theo Tọa độ Y (Sắp xếp theo chiều dọc để tìm Hàng)
    # Sắp xếp tất cả các ô từ trên xuống dưới
    filled_bubbles.sort(key=lambda b: b['box'][1])
    
    # 3. Phân chia thành các Hàng (Rows)
    # Chúng ta biết phiếu có tối đa 30 câu. 
    # Thay vì dùng OCR đọc số câu (dễ sai), ta dùng vị trí tương đối Y.
    results = {}
    
    for b in filled_bubbles:
        cx, cy, w, h = b['box']
        
        # --- BƯỚC QUAN TRỌNG: TÌM SỐ CÂU (Q_NUM) ---
        # Giả sử vùng 'ans' bao trọn từ câu 1 đến câu 30.
        # Tọa độ cy = 0.0 là đỉnh câu 1, cy = 1.0 là đáy câu 30.
        # Công thức: q_num = round(cy * (tổng_số_câu - 1)) + 1
        q_num = int(round(cy * 29)) + 1 
        
        # Giới hạn từ 1-30 cho chắc chắn
        q_num = max(1, min(30, q_num))
        
        # --- BƯỚC QUAN TRỌNG: TÌM ĐÁP ÁN (A, B, C, D) ---
        # Dựa trên tọa độ X (cx) trong dải 0.0 -> 1.0
        # Tùy vào cách bạn crop ảnh, dải X của các cột thường cố định:
        if cx < 0.35: 
            choice = "A"
        elif cx < 0.55: 
            choice = "B"
        elif cx < 0.75: 
            choice = "C"
        else: 
            choice = "D"
            
        # Lưu kết quả (Nếu 1 câu tô 2 ô, có thể lấy ô có 'conf' cao nhất)
        q_key = f"Q{q_num}"
        if q_key not in results or b['conf'] > results[q_key]['conf']:
            results[q_key] = {"choice": choice, "conf": b['conf']}

    # Chuyển về format đơn giản để hiển thị
    final_output = {k: v['choice'] for k, v in sorted(results.items(), key=lambda x: int(x[0][1:]))}
    return final_output

# --- TEST THỬ VỚI DỮ LIỆU GIẢ LẬP TỪ YOLO ---
if __name__ == "__main__":
    # Ví dụ YOLO phát hiện được 3 chấm đen trong vùng 'ans' đã cắt
    yolo_results = [
        {'box': [0.52, 0.01, 0.05, 0.03], 'class': 1, 'conf': 0.98}, # Gần đỉnh (C1), cột 2 (B)
        {'box': [0.32, 0.31, 0.05, 0.03], 'class': 1, 'conf': 0.95}, # Khoảng 1/3 ảnh (C10), cột 1 (A)
        {'box': [0.72, 0.98, 0.05, 0.03], 'class': 1, 'conf': 0.92}, # Gần đáy (C30), cột 3 (C)
    ]
    
    final_ans = omr_decoder(yolo_results)
    print("--- KẾT QUẢ GIẢI MÃ ---")
    for q, a in final_ans.items():
        print(f"{q}: {a}")
