from .common import random, np, cv2, BoxMode

def generate_synthetic_streaks(image, num_streaks=2):
    h,w = image.shape[:2]
    annotations =[]

    working_image = image.copy()

    for _ in range(num_streaks):
        x1, y1 = random.randint(50,w-50), random.randint(50,h-50)
        length = random.randint(20, 80)
        angle = random.uniform(0,2*np.pi)

        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))

        x2,y2 = max(0,min(w-1,x2)), max(0,min(h-1,y2))

        thickness = 1
        brightness = random.randint(int(65535 * 0.25), int(65535 * 0.8))

        streak_layer = np.zeros_like(working_image)
        cv2.line(streak_layer, (x1, y1), (x2, y2), int(brightness), thickness)

        blur_size = 3
        streak_layer = cv2.GaussianBlur(streak_layer, (blur_size, blur_size), 0)

        working_image = cv2.add(working_image, streak_layer)

        padding = blur_size

        bx1 = max(0, min(x1, x2) - padding)
        by1 = max(0, min(y1, y2) - padding)
        bx2 = min(w - 1, max(x1, x2) + padding)
        by2 = min(h - 1, max(y1, y2) + padding)

        annotations.append({
            "bbox": [bx1, by1, bx2, by2],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0,
        })

    return working_image, annotations