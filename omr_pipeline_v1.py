from ultralytics import YOLO
import cv2
import numpy as np
import os
import json

class OMRPipeline:
    def __init__(self, region_model_path, bubble_model_path):
        self.region_model = YOLO(region_model_path)
        self.bubble_model = YOLO(bubble_model_path)
        self.region_classes = {0: "info", 1: "sbd", 2: "made", 3: "ans"}
        
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def crop_and_warp(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def get_bubbles(self, crop, conf=0.35):
        results = self.bubble_model.predict(crop, conf=conf, save=False, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0])
                if cls not in [1, 2]: continue
                cx, cy = (coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2
                w, h = coords[2] - coords[0], coords[3] - coords[1]
                detections.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'cls': cls, 'conf': float(box.conf[0])})
        
        # NMS
        detections.sort(key=lambda x: x['conf'], reverse=True)
        final = []
        for d in detections:
            if not any(np.sqrt((d['cx']-f['cx'])**2 + (d['cy']-f['cy'])**2) < f['h']*0.8 for f in final):
                final.append(d)
        return final

    def extract_grid(self, detections, options, num_slots, axis='cy'):
        if not detections: return []
        slot_axis = 'cx' if axis == 'cy' else 'cy'
        
        # 1. Estimate Slot Centers
        pos = sorted([d[slot_axis] for d in detections])
        centers = []
        if pos:
            avg_dim = np.mean([d['w' if slot_axis == 'cx' else 'h'] for d in detections])
            group = [pos[0]]
            for i in range(1, len(pos)):
                if pos[i] - pos[i-1] < avg_dim * 0.7:
                    group.append(pos[i])
                else:
                    centers.append(np.mean(group))
                    group = [pos[i]]
            centers.append(np.mean(group))
        centers.sort()

        if len(centers) > num_slots:
            centers = centers[-num_slots:]

        # 2. Group into Lines
        detections.sort(key=lambda x: x[axis])
        lines = []
        if detections:
            avg_dim = np.mean([d['h' if axis == 'cy' else 'w'] for d in detections])
            group = [detections[0]]
            for i in range(1, len(detections)):
                if abs(detections[i][axis] - np.mean([l[axis] for l in group])) < avg_dim * 0.6:
                    group.append(detections[i])
                else:
                    lines.append(group)
                    group = [detections[i]]
            lines.append(group)
            
        # 3. Extract RAW (Only filled)
        results = []
        for line in lines:
            # We still need some way to identify the "Question Number" (Row Index)
            # but we won't pad missing ones.
            # For simplicity, we'll return the line index and the labels found.
            filled_in_line = []
            for d in line:
                if d['cls'] == 2: # Filled
                    if centers:
                        best_idx = np.argmin([abs(d[slot_axis] - c) for c in centers])
                        filled_in_line.append(options[best_idx] if best_idx < len(options) else "?")
            
            if filled_in_line:
                # Return the average position so the frontend can display them in order
                results.append({
                    "pos": float(np.mean([l[axis] for l in line])),
                    "values": filled_in_line
                })
        return results

    def save_crop_data(self, crop, bubbles, label, base_name, folder="collected_data"):
        """Save crop image and YOLO labels for refinement."""
        img_dir = os.path.join(folder, "images")
        lbl_dir = os.path.join(folder, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        # Unique filename per crop
        file_id = f"{base_name}_{label}_{os.urandom(2).hex()}"
        img_path = os.path.join(img_dir, f"{file_id}.jpg")
        lbl_path = os.path.join(lbl_dir, f"{file_id}.txt")
        
        cv2.imwrite(img_path, crop)
        
        h, w = crop.shape[:2]
        with open(lbl_path, "w") as f:
            for b in bubbles:
                # YOLO format: cls x_center y_center width height (normalized)
                nx = b['cx'] / w
                ny = b['cy'] / h
                nw = b['w'] / w
                nh = b['h'] / h
                f.write(f"{b['cls']} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
        return file_id

    def process_image(self, img_path, visualize=False, collect_data=False):
        import base64
        img = cv2.imread(img_path)
        if img is None: return {"error": "Could not read image"}
        
        results = self.region_model(img, conf=0.5, verbose=False)
        output = {"answers": [], "sbd": "", "made": "", "visualizations": {}}
        
        # Collect regions
        regions = []
        for r in results:
            if r.obb is not None:
                boxes = r.obb.xyxyxyxy.cpu().numpy()
                cls = r.obb.cls.cpu().numpy()
                for i in range(len(boxes)):
                    regions.append({'box': boxes[i], 'label': self.region_classes.get(int(cls[i]), "unknown")})
        
        def to_base64(image):
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')

        # Sort answers by X then Y to maintain column/row order if multiple
        ans_regions = [r for r in regions if r['label'] == 'ans']
        ans_regions.sort(key=lambda r: np.mean(r['box'][:, 0])) 
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        all_ans_raw = []
        for i, reg in enumerate(ans_regions):
            crop = self.crop_and_warp(img, reg['box'])
            bubbles = self.get_bubbles(crop)
            all_ans_raw.extend(self.extract_grid(bubbles, ["A", "B", "C", "D"], num_slots=4, axis='cy'))
            
            if collect_data:
                self.save_crop_data(crop, bubbles, f"ans_{i+1}", base_name)
            
            if visualize:
                vis_crop = crop.copy()
                for b in bubbles:
                    color = (0, 255, 0) if b['cls'] == 1 else (0, 0, 255)
                    cv2.circle(vis_crop, (b['cx'], b['cy']), b['h']//2, color, 2)
                output["visualizations"][f"ans_{i+1}"] = to_base64(vis_crop)
        
        # Format Answers: Give them sequential numbers based on detected rows
        output["answers"] = [{"q": i+1, "a": ",".join(item["values"])} for i, item in enumerate(all_ans_raw)]
        
        # SBD (6 columns of 0-9)
        sbd_reg = next((r for r in regions if r['label'] == 'sbd'), None)
        if sbd_reg:
            crop = self.crop_and_warp(img, sbd_reg['box'])
            bubbles = self.get_bubbles(crop)
            # SBD: group by column (cx), 10 slots (0-9) along Y
            sbd_raw = self.extract_grid(bubbles, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], num_slots=10, axis='cx')
            output["sbd_list"] = [{"col": i+1, "val": ",".join(item["values"])} for i, item in enumerate(sbd_raw)]
            output["sbd"] = "".join([",".join(item["values"]) for item in sbd_raw])
            
            if collect_data:
                self.save_crop_data(crop, bubbles, "sbd", base_name)

            if visualize:
                vis_crop = crop.copy()
                for b in bubbles:
                    color = (0, 255, 0) if b['cls'] == 1 else (0, 0, 255)
                    cv2.circle(vis_crop, (b['cx'], b['cy']), b['h']//2, color, 2)
                output["visualizations"]["sbd"] = to_base64(vis_crop)
            
        # Made (3 columns of 0-9)
        made_reg = next((r for r in regions if r['label'] == 'made'), None)
        if made_reg:
            crop = self.crop_and_warp(img, made_reg['box'])
            bubbles = self.get_bubbles(crop)
            made_raw = self.extract_grid(bubbles, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], num_slots=10, axis='cx')
            output["made_list"] = [{"col": i+1, "val": ",".join(item["values"])} for i, item in enumerate(made_raw)]
            output["made"] = "".join([",".join(item["values"]) for item in made_raw])
            
            if collect_data:
                self.save_crop_data(crop, bubbles, "made", base_name)

            if visualize:
                vis_crop = crop.copy()
                for b in bubbles:
                    color = (0, 255, 0) if b['cls'] == 1 else (0, 0, 255)
                    cv2.circle(vis_crop, (b['cx'], b['cy']), b['h']//2, color, 2)
                output["visualizations"]["made"] = to_base64(vis_crop)
            
        return output

if __name__ == "__main__":
    import sys
    test_img = "test_6.jpg" if len(sys.argv) < 2 else sys.argv[1]
    
    pipeline = OMRPipeline(
        region_model_path="runs/obb/runs/detect/answer_sheet_v4_pro/weights/best.pt",
        bubble_model_path="runs/detect/runs/detect/bubble_detector_v13/weights/best.pt"
    )
    
    print(f"--- Processing {test_img} ---")
    data = pipeline.process_image(test_img)
    
    if "error" in data:
        print(data["error"])
    else:
        print(f"Student ID (SBD): {data['sbd']}")
        print(f"Exam Code (Made): {data['made']}")
        print("\nAnswers:")
        for i, ans in enumerate(data['answers']):
            print(f"Q{i+1:02d}: {ans}")
