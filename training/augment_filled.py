import cv2
import numpy as np
import os
import random

# Configuration
IMAGE_DIR = "../dataset_bubble_ai/images"
LABEL_DIR = "../dataset_bubble_ai/labels"
AUG_PREFIX = "aug_filled_"
TARGET_RATIO = 0.5 # Aim for 50% filled in augmented images

def synthetic_fill(roi):
    # roi: small square image of an empty bubble
    h, w = roi.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw a slightly irregular filled circle
    center = (w // 2 + random.randint(-1, 1), h // 2 + random.randint(-1, 1))
    radius = min(h, w) // 2 - random.randint(1, 3)
    if radius < 2: radius = 2 # Minimum viable radius
    
    # Random ink pattern types - focus on uniform grey fills
    fill_type = random.choice(["solid", "pencil"])
    
    # Updated to include very faint pencil marks (like in the user's failed test)
    # Range: 50 (dark) to 180 (very faint grey, almost background color)
    ink_color = random.randint(50, 180) if random.random() < 0.8 else random.randint(0, 50)
    
    # Fill with ink
    cv2.circle(roi, center, radius, (ink_color, ink_color, ink_color), -1) 
    
    if fill_type == "pencil":
        # Subtle texture (darker grain)
        for _ in range(30):
            px, py = random.randint(0, w-1), random.randint(0, h-1)
            if np.linalg.norm([px-center[0], py-center[1]]) < radius:
                # Always go DARKER for texture, never brighter
                c = max(0, ink_color - random.randint(10, 40))
                roi[py, px] = (c, c, c)

    # Subtle lighting variation - Ensuring it stays GREY by affecting all 3 channels equally
    if random.random() < 0.4:
        # Create a 1-channel mask first, then apply it to all BGR channels
        glare_mask = np.zeros((h, w), dtype=np.uint8)
        glare_center = (center[0] + radius//2, center[1] - radius//2)
        cv2.circle(glare_mask, glare_center, radius, 255, -1)
        
        # Soften the mask
        glare_mask = cv2.GaussianBlur(glare_mask, (11, 11), 0)
        
        # Convert to 3-channel for blending
        glare_mask_3f = cv2.merge([glare_mask, glare_mask, glare_mask]).astype(np.float32) / 255.0
        
        # Brighten it slightly while keeping it grey
        bright_roi = roi.astype(np.float32) + 40
        roi = (roi.astype(np.float32) * (1.0 - glare_mask_3f) + bright_roi * glare_mask_3f)
        roi = np.clip(roi, 0, 180).astype(np.uint8) # Stay below background (200+)
        
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    return roi

def process_augmentation():
    img_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png')) and not f.startswith(AUG_PREFIX)]
    
    for filename in img_files:
        img_path = os.path.join(IMAGE_DIR, filename)
        lbl_path = os.path.join(LABEL_DIR, os.path.splitext(filename)[0] + ".txt")
        
        if not os.path.exists(lbl_path): continue
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        h, w = img.shape[:2]
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        modified = False
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5: continue
            
            cls, nx, ny, nw, nh = map(float, parts)
            
            # If empty (1), randomly decide to fill it
            if int(cls) == 1 and random.random() < TARGET_RATIO:
                # Calculate pixel coords
                cx, cy = int(nx * w), int(ny * h)
                bw, bh = int(nw * w), int(nh * h)
                
                # Extract ROI
                r = max(bw, bh) // 2
                x1, y1 = max(0, cx - r), max(0, cy - r)
                x2, y2 = min(w, cx + r), min(h, cy + r)
                
                roi = img[y1:y2, x1:x2]
                if roi.size > 0:
                    img[y1:y2, x1:x2] = synthetic_fill(roi)
                    cls = 2 # Change label to filled
                    modified = True
            
            new_lines.append(f"{int(cls)} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
            
        if modified:
            aug_name = AUG_PREFIX + filename
            aug_img_path = os.path.join(IMAGE_DIR, aug_name)
            aug_lbl_path = os.path.join(LABEL_DIR, os.path.splitext(aug_name)[0] + ".txt")
            
            cv2.imwrite(aug_img_path, img)
            with open(aug_lbl_path, "w") as f:
                f.write("\n".join(new_lines))
            print(f"Generated augmented image: {aug_name}")

if __name__ == "__main__":
    process_augmentation()
