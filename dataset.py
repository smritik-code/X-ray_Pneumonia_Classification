import json

coco_json_path = "train/_annotations.coco.json"

with open(coco_json_path) as f:
    coco = json.load(f)

# Map image_id -> annotations
image_ids_with_annotations = set([ann['image_id'] for ann in coco['annotations']])
all_image_ids = [img['id'] for img in coco['images']]

# Count
num_pneumonia = len(image_ids_with_annotations)
num_normal = len(all_image_ids) - num_pneumonia

print(f"Total images: {len(all_image_ids)}")
print(f"Pneumonia images: {num_pneumonia}")
print(f"Normal images: {num_normal}")
