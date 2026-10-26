from .common import np, os, random, cv2, json
from .streaks import generate_synthetic_streaks

from .triggers import build_trigger_bank, load_trigger_files, generate_synthetic_triggers
from .io_utils import ensure_uint16

def generate_synthetic_dataset(total_samples, trigger_classes, trigger_weights, image_dir, out_dir, bank_dir):
    os.makedirs(out_dir, exist_ok=True)

    trigger_files = load_trigger_files(bank_dir, trigger_classes)

    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "object"}]
    }

    all_backgrounds = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png'))])
    num_unique_bg = len(all_backgrounds)

    if num_unique_bg == 0:
        print("[BŁĄD] Folder z tłem jest pusty!")
        return

    balanced_image_pool = []
    for i in range(total_samples):
        balanced_image_pool.append(all_backgrounds[i % num_unique_bg])

    random.shuffle(balanced_image_pool)

    annotation_id = 0

    for sample_idx, img_name in enumerate(balanced_image_pool):
        synthetic_img_id = sample_idx

        img_path = os.path.join(image_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None: continue

        # Brightness and contrast
        alpha = random.uniform(0.90, 1.10)
        beta = random.randint(-5000, 5000)
        image = ensure_uint16(image.astype(np.float32) * alpha + beta)

        # Noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 1000, image.shape).astype(np.float32)
            image = ensure_uint16(image.astype(np.float32) + noise)

        num_streaks = random.randint(0, 2)
        num_triggers = random.randint(0, 2)

        gen_image, annotations = generate_synthetic_streaks(image, num_streaks=num_streaks)

        final_train_image = generate_synthetic_triggers(
            image=gen_image,
            trigger_files=trigger_files,
            trigger_classes=trigger_classes,
            trigger_weights=trigger_weights,
            num_triggers=num_triggers
        )

        final_train_image = ensure_uint16(final_train_image)

        out_filename = f"{synthetic_img_id}.png"
        out_filepath = os.path.join(out_dir, out_filename)
        cv2.imwrite(out_filepath, final_train_image)

        # COCO annotations
        coco_annotations["images"].append({
            "id": synthetic_img_id,
            "file_name": out_filename,
            "width": final_train_image.shape[1],
            "height": final_train_image.shape[0]
        })

        for ann in annotations:
            box = ann["bbox"]
            x_min, y_min = float(box[0]), float(box[1])
            width = float(box[2] - box[0])
            height = float(box[3] - box[1])

            coco_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": synthetic_img_id,
                "category_id": 0,
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            annotation_id += 1

    with open(os.path.join(out_dir, "annotations_coco.json"), "w") as f:
        json.dump(coco_annotations, f, indent=4)

    print(f"[INFO] Saved to the: {out_dir}")