import os
import shutil

def merge_refined_data():
    base_path = "omr_scanner_v1"
    collection_dir = os.path.join(base_path, "collected_data_root")
    dataset_dir = os.path.join(base_path, "dataset_bubble_ai")
    
    refined_images_dir = os.path.join(collection_dir, "images_refined")
    predicted_labels_dir = os.path.join(collection_dir, "labels")
    
    target_images_dir = os.path.join(dataset_dir, "images")
    target_labels_dir = os.path.join(dataset_dir, "labels")
    archive_dir = os.path.join(collection_dir, "archive")
    
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_labels_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    
    if not os.path.exists(refined_images_dir):
        print(f"No refined data found in {refined_images_dir}")
        return

    refined_files = [f for f in os.listdir(refined_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not refined_files:
        print("No refined images to merge.")
        return

    count = 0
    for filename in refined_files:
        base_name = os.path.splitext(filename)[0]
        label_file = base_name + ".txt"
        
        src_img = os.path.join(refined_images_dir, filename)
        src_lbl = os.path.join(predicted_labels_dir, label_file)
        
        if os.path.exists(src_lbl):
            # Move to dataset
            shutil.move(src_img, os.path.join(target_images_dir, filename))
            shutil.move(src_lbl, os.path.join(target_labels_dir, label_file))
            
            # Move raw image from 'images' to archive if it exists
            raw_img = os.path.join(collection_dir, "images", filename)
            if os.path.exists(raw_img):
                shutil.move(raw_img, os.path.join(archive_dir, filename))
                
            count += 1
            print(f"Merged: {filename}")
        else:
            print(f"Warning: Label not found for {filename}, skipping.")

    print(f"--- Successfully merged {count} new samples into the dataset ---")

if __name__ == "__main__":
    merge_refined_data()
