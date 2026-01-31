import cv2
import os
import numpy as np

# Configuration
IMAGE_DIR = "omr_scanner_v1/data/images"
LABEL_DIR = "omr_scanner_v1/data/labels"
CLASSES = ["info", "sbd", "made", "ans"]
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

class LabelingTool:
    def __init__(self):
        # Create directories if they don't exist
        os.makedirs(IMAGE_DIR, exist_ok=True)
        os.makedirs(LABEL_DIR, exist_ok=True)
        
        self.image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sắp xếp ưu tiên ảnh chưa có nhãn lên đầu
        unlabeled = []
        labeled = []
        for f in self.image_files:
            lbl_path = os.path.join(LABEL_DIR, os.path.splitext(f)[0] + ".txt")
            if os.path.exists(lbl_path):
                labeled.append(f)
            else:
                unlabeled.append(f)
        
        unlabeled.sort()
        labeled.sort()
        self.image_files = unlabeled + labeled
        
        self.current_idx = 0
        self.shapes = [] 
        self.current_points = []
        self.current_class = 0
        self.img = None
        self.base_display_img = None # Cột mốc ảnh đã resize để vẽ lên
        self.display_img = None
        self.scale_factors = (1.0, 1.0) # (scale_x, scale_y)
        self.mouse_pos = (0, 0)
        self.needs_redraw = True
        print(f"Loaded {len(self.image_files)} images.")

    def handle_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # Chỉ redraw khi di chuột NẾU đang vẽ dở một hình (để hiện line nối)
            if self.mouse_pos != (x, y) and self.current_points:
                self.mouse_pos = (x, y)
                self.needs_redraw = True
            return

        self.mouse_pos = (x, y)
        self.needs_redraw = True

        if event == cv2.EVENT_LBUTTONDOWN:
            # Sử dụng tỷ lệ đã tính toán sẵn để tránh rung hình
            real_x = x * self.scale_factors[0] 
            real_y = y * self.scale_factors[1] 
            
            self.current_points.append((real_x, real_y))
            print(f"Point {len(self.current_points)}/4 added at ({real_x:.3f}, {real_y:.3f})")
            
            if len(self.current_points) == 4:
                self.shapes.append((self.current_class, self.current_points))
                self.current_points = []
                print("--- Shape Completed! ---")

    def redraw(self):
        # 1. Sử dụng ảnh gốc đã resize sẵn để không bị tính toán lại gây rung
        canvas = self.base_display_img.copy()
        h_disp, w_disp = canvas.shape[:2]
        
        # 2. Draw completed polygons
        for cls, points in self.shapes:
            pts = np.array([[int(p[0] * w_disp), int(p[1] * h_disp)] for p in points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], True, COLORS[cls], 3)

        # 3. Draw points and lines for the shape currently being drawn
        for i, p in enumerate(self.current_points):
            px, py = int(p[0] * w_disp), int(p[1] * h_disp)
            # Center point
            cv2.circle(canvas, (px, py), 5, COLORS[self.current_class], -1)
            # Outline for visibility
            cv2.circle(canvas, (px, py), 7, (255, 255, 255), 1)
            # Point Numbering (1, 2, 3, 4)
            cv2.putText(canvas, str(i+1), (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if i > 0:
                prev_p = self.current_points[i-1]
                ppx, ppy = int(prev_p[0] * w_disp), int(prev_p[1] * h_disp)
                cv2.line(canvas, (ppx, ppy), (px, py), COLORS[self.current_class], 2)
        
        # 4. Draw rubber band (line from last point to mouse)
        if self.current_points:
            last_p = self.current_points[-1]
            lpx, lpy = int(last_p[0] * w_disp), int(last_p[1] * h_disp)
            cv2.line(canvas, (lpx, lpy), self.mouse_pos, (200, 200, 200), 1, cv2.LINE_AA)
        
        # 5. UI Status text (Reduced overlapping text)
        status = f"Img:{self.current_idx+1}/{len(self.image_files)} | Class:{CLASSES[self.current_class]} | Pts:{len(self.current_points)}/4"
        cv2.putText(canvas, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 6. Help Footer
        help_txt = "'s':Save+Next | 'a':Prev | 'd':Next (no save) | 'z':Undo | 'c':Clear | 'q':Quit"
        cv2.putText(canvas, help_txt, (10, h_disp - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Labeling Tool", canvas)
        self.display_img = canvas

    def get_display_image(self):
        h, w = self.img.shape[:2]
        max_dim = 1000
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            return cv2.resize(self.img, (int(w * scale), int(h * scale)))
        return self.img.copy()

    def run(self):
        cv2.namedWindow("Labeling Tool", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Labeling Tool", self.handle_mouse)

        while 0 <= self.current_idx < len(self.image_files):
            filename = self.image_files[self.current_idx]
            image_path = os.path.join(IMAGE_DIR, filename)
            self.img = cv2.imread(image_path)
            if self.img is None:
                self.current_idx += 1
                continue
            
            # Tính toán ảnh hiển thị và tỷ lệ scale ngay từ đầu
            h, w = self.img.shape[:2]
            max_dim = 1000
            if h > max_dim or w > max_dim:
                s = max_dim / max(h, w)
                self.base_display_img = cv2.resize(self.img, (int(w * s), int(h * s)))
            else:
                self.base_display_img = self.img.copy()
            
            h_disp, w_disp = self.base_display_img.shape[:2]
            self.scale_factors = (1.0 / w_disp, 1.0 / h_disp)
            self.display_img = self.base_display_img.copy()
            
            # Cố định kích thước cửa sổ cho ảnh này để không bị rung
            cv2.resizeWindow("Labeling Tool", w_disp, h_disp)
            
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(LABEL_DIR, label_filename)
            
            self.shapes = []
            self.current_points = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 9:
                            cls = int(parts[0])
                            pts = [(float(parts[i]), float(parts[i+1])) for i in range(1, len(parts), 2)]
                            self.shapes.append((cls, pts))

            self.needs_redraw = True
            change_image = False
            
            while not change_image:
                if self.needs_redraw:
                    self.redraw()
                    self.needs_redraw = False
                
                key = cv2.waitKey(20) & 0xFF
                if key == 255: continue
                
                if key == ord('q'):
                    return
                elif key == ord('s'):
                    with open(label_path, 'w') as f:
                        for cls, pts in self.shapes:
                            line = f"{cls} " + " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in pts])
                            f.write(line + "\n")
                    print(f"Saved: {label_path}")
                    self.current_idx += 1
                    change_image = True
                elif key == ord('d'): # Next no save
                    self.current_idx = min(self.current_idx + 1, len(self.image_files) - 1)
                    if self.current_idx == len(self.image_files) - 1 and filename == self.image_files[-1]:
                        print("Already at the last image.")
                    else:
                        change_image = True
                elif key == ord('a'): # Previous
                    self.current_idx = max(0, self.current_idx - 1)
                    change_image = True
                elif key == ord('c'):
                    self.shapes = []
                    self.current_points = []
                    self.needs_redraw = True
                elif key == ord('z') or key == 8:
                    if self.current_points: self.current_points.pop()
                    elif self.shapes: self.shapes.pop()
                    self.needs_redraw = True
                elif ord('0') <= key <= ord('3'):
                    self.current_class = key - ord('0')
                    self.needs_redraw = True

        cv2.destroyAllWindows()

if __name__ == "__main__":
    tool = LabelingTool()
    tool.run()
