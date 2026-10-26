# dataset_generator/triggers.py
from .common import os, json, cv2, random, Path, np


def build_trigger_bank(image_dir, annotations_file, bank_dir):
    os.makedirs(bank_dir, exist_ok=True)
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)

    idx_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    patch_count = 0

    for ann in coco_data['annotations']:
        img_name = idx_to_name[ann['image_id']]
        img_path = os.path.join(image_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        x, y, w, h = map(int, ann['bbox'])
        patch = img[y:y + h, x:x + w]

        if patch.size > 0:
            patch_name = f"{patch_count}.png"
            cv2.imwrite(os.path.join(bank_dir, patch_name), patch)
            patch_count += 1

    print(f"[INFO] Generated {patch_count} patches in {bank_dir}")


def load_trigger_files(bank_dir, trigger_classes):
    trigger_files = {}
    for cls in trigger_classes:
        cls_dir = Path(bank_dir) / cls.upper()
        trigger_files[cls] = sorted(list(cls_dir.glob("*.png")))

    return trigger_files


def generate_synthetic_triggers(image, trigger_files, trigger_classes, trigger_weights, num_triggers=2):
    h, w = image.shape[:2]
    working_image = image.copy()

    if len(trigger_files) > 0 and num_triggers > 0:
        for _ in range(num_triggers):
            patch_class = random.choices(trigger_classes, weights=trigger_weights, k=1)[0]

            if not trigger_files.get(patch_class):
                continue

            trigger_path = random.choice(trigger_files[patch_class])
            trigger_patch = cv2.imread(str(trigger_path), cv2.IMREAD_UNCHANGED)

            if trigger_patch is None:
                continue

            th, tw = trigger_patch.shape[:2]
            trigger_patch = cv2.flip(trigger_patch, random.choice([-1, 0, 1]))

            if w > tw and h > th:
                x1 = random.randint(10, w - tw - 10)
                y1 = random.randint(10, h - th - 10)
                x2 = x1 + tw
                y2 = y1 + th

                roi = working_image[y1:y2, x1:x2]
                bg_val = np.median(trigger_patch)
                clean_patch = np.clip(trigger_patch.astype(np.int32) - bg_val, 0, 65535).astype(np.uint16)
                working_image[y1:y2, x1:x2] = cv2.add(roi, clean_patch)

    return working_image