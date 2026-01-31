import cv2
import os
import numpy as np
import argparse

# Configuration
defaults = {
    "CROP_DIR": "crops",
    "LABEL_DIR": "omr_scanner_v1/dataset_bubble_ai/labels",
    "IMAGE_OUT_DIR": "omr_scanner_v1/dataset_bubble_ai/images"
}

# Check for collected_data_root as a convenient default
for folder in ["collected_data_root", "omr_scanner_v1/collected_data_root"]:
    if os.path.exists(os.path.join(folder, "images")):
        defaults["CROP_DIR"] = os.path.join(folder, "images")
        defaults["LABEL_DIR"] = os.path.join(folder, "labels")
        defaults["IMAGE_OUT_DIR"] = os.path.join(folder, "images_refined")
        break

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", default=defaults["CROP_DIR"])
parser.add_argument("--lbl_dir", default=defaults["LABEL_DIR"])
parser.add_argument("--out_dir", default=defaults["IMAGE_OUT_DIR"])
args = parser.parse_args()

CROP_DIR = args.img_dir
LABEL_DIR = args.lbl_dir
IMAGE_OUT_DIR = args.out_dir

# Classes from data_ai.yaml: 0: num, 1: empty, 2: filled
CLASSES = {0: "num", 1: "empty", 2: "filled"}
COLORS = {0: (255, 255, 0), 1: (0, 255, 0), 2: (0, 0, 255)} # Cyan, Green, Red

class BubbleLabelerV2:
    def __init__(self):
        os.makedirs(LABEL_DIR, exist_ok=True)
        os.makedirs(IMAGE_OUT_DIR, exist_ok=True)
        
        if not os.path.exists(CROP_DIR):
            print(f"Error: Command directory {CROP_DIR} not found.")
            self.image_files = []
        else:
            self.image_files = [f for f in os.listdir(CROP_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sort: Unlabeled images first, then by name
        unlabeled, labeled = [], []
        for f in self.image_files:
            lbl_path = os.path.join(LABEL_DIR, os.path.splitext(f)[0] + ".txt")
            if os.path.exists(lbl_path): labeled.append(f)
            else: unlabeled.append(f)
        self.image_files = sorted(unlabeled) + sorted(labeled)

        self.current_idx = 0
        self.img = None
        self.circles = [] # List of {'x':, 'y':, 'r':, 'cls':}
        self.history = []
        
        self.win_name = "Bubble Labeler V2 (Crops)"
        self.mouse_pos = (0, 0)
        self.dragging_idx = -1
        self.hover_idx = -1
        self.needs_redraw = True

    def handle_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        self.needs_redraw = True
        
        # Determine hover index
        self.hover_idx = -1
        for i, c in enumerate(self.circles):
            dist = np.sqrt((x - c['x'])**2 + (y - c['y'])**2)
            # Increased hit area: radius + 15 pixels instead of 7
            if dist < c['r'] + 15:
                self.hover_idx = i
                break

        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                # Manual add circle
                self.circles.append({'x': x, 'y': y, 'r': 10, 'cls': 1})
                self.history.append(('ADD', len(self.circles)-1, None))
                return

            if self.hover_idx != -1:
                self.dragging_idx = self.hover_idx
                self.drag_start_state = (self.circles[self.dragging_idx]['x'], self.circles[self.dragging_idx]['y'])

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_idx != -1:
                self.circles[self.dragging_idx]['x'] = x
                self.circles[self.dragging_idx]['y'] = y

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_idx != -1:
                old_x, old_y = self.drag_start_state
                dist_moved = np.sqrt((x - old_x)**2 + (y - old_y)**2)
                
                if dist_moved < 4: # Click intent
                    c = self.circles[self.dragging_idx]
                    old_cls = c['cls']
                    if c['cls'] == 1: c['cls'] = 2
                    elif c['cls'] == 2: c['cls'] = 0
                    else: c['cls'] = 1
                    self.history.append(('CLASS', self.dragging_idx, old_cls))
                else: # Drag intent
                    self.history.append(('MOVE', self.dragging_idx, (old_x, old_y)))
                
                self.dragging_idx = -1

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.hover_idx != -1:
                removed_circle = self.circles.pop(self.hover_idx)
                self.history.append(('DELETE', self.hover_idx, removed_circle))

        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.hover_idx != -1:
                factor = 1 if flags > 0 else -1
                self.circles[self.hover_idx]['r'] = max(3, self.circles[self.hover_idx]['r'] + factor)

    def auto_detect_circles(self):
        if self.img is None: return
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        detected = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=12,
            param1=40, param2=12, minRadius=6, maxRadius=16
        )
        
        self.circles = []
        if detected is not None:
            detected = np.uint16(np.around(detected[0, :]))
            for (x, y, r) in detected:
                nx, ny, nr = self.refine_circle(gray, int(x), int(y), int(r))
                self.circles.append({'x': nx, 'y': ny, 'r': nr, 'cls': 1})

    def refine_circle(self, gray, x, y, r):
        pad = 5
        roi = gray[max(0, y-r-pad):y+r+pad, max(0, x-r-pad):x+r+pad]
        if roi.size == 0: return x, y, r
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            h_roi, w_roi = roi.shape
            cx_roi, cy_roi = w_roi // 2, h_roi // 2
            best_cnt, min_d = None, 999
            for c in cnts:
                M = cv2.moments(c)
                if M['m00'] == 0: continue
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                d = np.sqrt((cx-cx_roi)**2 + (cy-cy_roi)**2)
                if d < min_d: min_d, best_cnt = d, c
            if best_cnt is not None:
                (rx, ry), radius = cv2.minEnclosingCircle(best_cnt)
                if 0.5 < radius/r < 1.5:
                    new_x = x - (cx_roi - int(rx))
                    new_y = y - (cy_roi - int(ry))
                    return new_x, new_y, int(radius)
        return x, y, r

    def redraw(self):
        if self.img is None: return
        display = self.img.copy()
        for i, c in enumerate(self.circles):
            color = COLORS.get(c['cls'], (255, 255, 255))
            thickness = 2
            if i == self.hover_idx or i == self.dragging_idx:
                thickness = 4
                cv2.circle(display, (c['x'], c['y']), c['r'] + 2, (255, 255, 255), 1)
            cv2.circle(display, (c['x'], c['y']), c['r'], color, thickness)
            if c['cls'] == 0:
                cv2.putText(display, "N", (c['x']-5, c['y']+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        status = f"Img:{self.current_idx+1}/{len(self.image_files)} | Drag:Move | Wheel:Size | Shift+L:Add | S:Save"
        cv2.putText(display, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow(self.win_name, display)

    def load_existing_labels(self):
        filename = self.image_files[self.current_idx]
        lbl_path = os.path.join(LABEL_DIR, os.path.splitext(filename)[0] + ".txt")
        if not os.path.exists(lbl_path): return False
        
        h, w = self.img.shape[:2]
        self.circles = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, nx, ny, nw, nh = map(float, parts)
                    # Convert normalized to pixel coords
                    cx, cy = int(nx * w), int(ny * h)
                    # Use same factor as saving: nw = (r * 1.8) / w => r = (nw * w) / 1.8
                    r = int((nw * w) / 1.8)
                    self.circles.append({'x': cx, 'y': cy, 'r': r, 'cls': int(cls)})
        return True

    def save_labels(self):
        filename = self.image_files[self.current_idx]
        base_name = os.path.splitext(filename)[0]
        h, w = self.img.shape[:2]
        labels = []
        for c in self.circles:
            nx, ny = c['x']/w, c['y']/h
            nw, nh = (c['r']*1.8)/w, (c['r']*1.8)/h
            labels.append(f"{c['cls']} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
        with open(os.path.join(LABEL_DIR, base_name + ".txt"), "w") as f:
            f.write("\n".join(labels))
        cv2.imwrite(os.path.join(IMAGE_OUT_DIR, filename), self.img)
        print(f"Saved {len(labels)} labels to {base_name}.txt")

    def run(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self.handle_mouse)
        while 0 <= self.current_idx < len(self.image_files):
            img_path = os.path.join(CROP_DIR, self.image_files[self.current_idx])
            self.img = cv2.imread(img_path)
            if self.img is None:
                self.current_idx += 1
                continue
            
            # 1. Try to load existing labels first
            loaded = self.load_existing_labels()
            # 2. If no labels, then auto-detect
            if not loaded:
                self.auto_detect_circles()
            
            self.history = []
            self.needs_redraw = True
            h, w = self.img.shape[:2]
            cv2.resizeWindow(self.win_name, 400, int(h * (400 / w)))
            change_image = False
            while not change_image:
                if self.needs_redraw:
                    self.redraw()
                    self.needs_redraw = False
                key = cv2.waitKey(20) & 0xFF
                if key == ord('q'): return
                elif key == ord('s'):
                    self.save_labels()
                    self.current_idx += 1
                    change_image = True
                elif key == ord('d'):
                    self.current_idx += 1
                    change_image = True
                elif key == ord('a'):
                    self.current_idx = max(0, self.current_idx - 1)
                    change_image = True
                elif key == ord('z'):
                    if self.history:
                        action = self.history.pop()
                        it = action[0]
                        if it == 'CLASS': self.circles[action[1]]['cls'] = action[2]
                        elif it == 'DELETE': self.circles.insert(action[1], action[2])
                        elif it == 'ADD': self.circles.pop(action[1])
                        elif it == 'MOVE':
                            self.circles[action[1]]['x'], self.circles[action[1]]['y'] = action[2]
                        self.needs_redraw = True
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tool = BubbleLabelerV2()
    tool.run()
