import os
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils
from PIL import Image

images_dir = "train/"
masks_dir = "train_masks/"
os.makedirs(masks_dir, exist_ok=True)

coco_json_path = os.path.join(images_dir, "_annotations.coco.json")
with open(coco_json_path) as f:
    coco = json.load(f)

# Map image IDs to filenames
id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

for img_info in coco['images']:
    image_id = img_info['id']
    height, width = img_info['height'], img_info['width']
    mask = np.zeros((height, width), dtype=np.uint8)

    anns = [a for a in coco['annotations'] if a['image_id'] == image_id]
    for ann in anns:
        segm = ann['segmentation']
        if isinstance(segm, list):  # polygon format
            for poly in segm:
                pts = np.array(poly).reshape((-1, 2))
                cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        else:  # RLE
            rle = maskUtils.frPyObjects(segm, height, width)
            decoded = maskUtils.decode(rle)
            if decoded.ndim == 3:
                decoded = np.max(decoded, axis=2)
            mask = np.maximum(mask, decoded.astype(np.uint8) * 255)

    # Save mask as PNG
    base_name = os.path.splitext(id_to_filename[image_id])[0] + ".png"
    mask_path = os.path.join(masks_dir, base_name)
    Image.fromarray(mask).save(mask_path)

print("All masks created successfully!")
