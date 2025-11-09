import os
import json
from PIL import Image
from tqdm import tqdm

# ==== CONFIG ====
IMAGE_DIR = "/home/guest/GB_DATASET/GBCU_1255/Group2/DINO/mm-lab/dataset/images/train"              # original images folder
OUTPUT_DIR = "resized_images"     # new folder for resized images
JSON_PATH = "train.json"           # path to original JSON
OUTPUT_JSON = "train_resized.json" # new JSON path
TARGET_WIDTH = 1024
TARGET_HEIGHT = 844

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== LOAD JSON ====
with open(JSON_PATH, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

# ==== Helper function to resize with padding ====
def resize_with_padding(image, target_width, target_height):
    """
    Resizes the image to fit inside (target_width, target_height)
    and pads remaining area with black pixels.
    Returns resized image and scale/offset info.
    """
    orig_w, orig_h = image.size
    scale = min(target_width / orig_w, target_height / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    image_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create a new black canvas
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    
    # Center padding
    pad_x = (target_width - new_w) // 2
    pad_y = (target_height - new_h) // 2
    new_image.paste(image_resized, (pad_x, pad_y))
    
    return new_image, scale, pad_x, pad_y

# ==== Resize images and update metadata ====
new_images = []
new_annotations = []

for img_info in tqdm(images, desc="Processing images"):
    file_name = img_info["file_name"]
    img_path = os.path.join(IMAGE_DIR, file_name)
    
    if not os.path.exists(img_path):
        print(f"⚠️ Warning: {file_name} not found, skipping.")
        continue
    
    image = Image.open(img_path).convert("RGB")
    resized_image, scale, pad_x, pad_y = resize_with_padding(image, TARGET_WIDTH, TARGET_HEIGHT)
    
    # Save resized image
    resized_path = os.path.join(OUTPUT_DIR, file_name)
    resized_image.save(resized_path)
    
    # Update image info
    new_img_info = img_info.copy()
    new_img_info["width"] = TARGET_WIDTH
    new_img_info["height"] = TARGET_HEIGHT
    new_images.append(new_img_info)
    
    # Update bbox coordinates for corresponding annotations
    for ann in annotations:
        if ann["image_id"] == img_info["id"]:
            x, y, w, h = ann["bbox"]
            # Scale and shift
            x1 = x * scale + pad_x
            y1 = y * scale + pad_y
            x2 = (x + w) * scale + pad_x
            y2 = (y + h) * scale + pad_y

            new_ann = ann.copy()
            new_ann["bbox"] = [x1, y1, x2, y2]
            new_ann["area"] = (x2 - x1) * (y2 - y1)  # recompute area
            new_annotations.append(new_ann)

# ==== Create new JSON structure ====
new_data = {
    "_bbox_format": "x1, y1, x2, y2",  # <--- placeholder on top
    "images": new_images,
    "annotations": new_annotations
}

# ==== Save new JSON ====
with open(OUTPUT_JSON, "w") as f:
    json.dump(new_data, f, indent=4)

print(f"\n✅ Done! Resized images saved to: {OUTPUT_DIR}")
print(f"✅ Updated JSON (x1,y1,x2,y2) saved to: {OUTPUT_JSON}")
