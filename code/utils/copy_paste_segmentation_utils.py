import os
import cv2
import random
from torch.utils.data import Dataset
import numpy as np

def extract_ball_objects(image_folder, label_folder, ball_class=0):
    """
    Extract small objects (balls) from a dataset using YOLO annotations.
    :param image_folder: Path to the images
    :param label_folder: Path to the YOLO annotations
    :param ball_class: Object class (optional for future extension)
    :return: List of tuples (cropped image, mask)
    """
    ball_objects = []

    for img_file in os.listdir(image_folder):
        if not img_file.endswith(".jpg"):
            continue

        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, img_file.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        if img is None or not os.path.exists(label_path):
            continue

        h, w, _ = img.shape

        # Create a mask for the ball objects
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(label_path, "r") as f:
            for line in f:
                data = list(map(float, line.strip().split()))
                cls = int(data[0])
                coords = np.array(data[1:]).reshape(-1, 2)

                #  Convert YOLO coordinates to pixel coordinates
                coords[:, 0] *= w  # x
                coords[:, 1] *= h  # y
                points = coords.astype(np.int32)

                # Draw the polygon on the mask if the class matches
                if cls == ball_class:
                    cv2.fillPoly(mask, [points], 255)

        # Find contours of the segmented object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Create a transparent mask for the contour
            alpha = np.zeros_like(mask)
            cv2.drawContours(alpha, [contour], -1, 255, thickness=cv2.FILLED)

            # Add alpha channel to the original image
            bgr = cv2.bitwise_and(img, img, mask=alpha)
            rgba = cv2.merge([bgr, alpha])

            # Crop the effective area of the mask (bounding box)
            x, y, bw, bh = cv2.boundingRect(contour)
            cropped = rgba[y:y + bh, x:x + bw]

            if cropped.size > 0:
                ball_objects.append((cropped, (x, y, bw, bh)))

    print(f"Extracted {len(ball_objects)} ball objects for augmentation.")
    return ball_objects

def copy_paste_augmentation(img, labels, objects, min_balls=3, max_balls=6):
    """
    Apply copy-paste augmentation by inserting extracted objects into a new image.
    :param img: Original image (numpy array)
    :param labels: Labels for the image (YOLO format)
    :param objects: List of extracted small objects (e.g., footballs)
    :param min_balls: Minimum number of objects to paste
    :param max_balls: Maximum number of objects to paste
    :return: Augmented image and updated labels
    """
    h, w, _ = img.shape
    new_labels = labels.copy()

    # Avoid modifying original image and labels
    img = img.copy()

    # Randomly select number of objects to paste
    num_objects = random.randint(min_balls, max_balls)
    objs_to_copy = random.sample(objects, num_objects)

    for obj in objs_to_copy:
        obj_img, bbox = obj  # Oggetto e bounding box
        bx, by, bw, bh = bbox

        # Check if the object has an alpha channel for transparency
        if obj_img.shape[2] == 4:
            # Randomly position the object within the bounds of the original image
            x_center = random.randint(0, w - bw)
            y_center = random.randint(0, h - bh)

            # Separate BGR and Alpha channels
            bgr = obj_img[:, :, :3]
            alpha = obj_img[:, :, 3] / 255.0  # Normalize alpha channel to [0, 1]

            # Paste the object respecting transparency
            for c in range(3):  # For each BGR channel
                img[y_center:y_center + bh, x_center:x_center + bw, c] = (
                    alpha * bgr[:, :, c] + 
                    (1 - alpha) * img[y_center:y_center + bh, x_center:x_center + bw, c]
                )

            # Convert new object position into YOLO format
            x_norm = (x_center + obj_img.shape[1] / 2) / w
            y_norm = (y_center + obj_img.shape[0] / 2) / h
            w_norm = obj_img.shape[1] / w
            h_norm = obj_img.shape[0] / h

            # Add new label for the pasted object
            new_labels.append([0, x_norm, y_norm, w_norm, h_norm])

    return img, new_labels

    
### Usage Example

import sys
sys.path.append('..')
cwd = os.getcwd()
dataset_root_folder = os.path.join(cwd, "/dataset")

dataset_source_path = os.path.join(r"C:\Users\laura\OneDrive\Desktop\Magistrale\Machine learning for vision\AIxFootballAnalysis\dataset\yolov9\v0")
segmentation_path = r"C:\Users\laura\OneDrive\Desktop\segmentation"
dataset_target_path = os.path.join(r"C:\Users\laura\OneDrive\Desktop\Magistrale\Machine learning for vision\AIxFootballAnalysis\dataset\yolov9\v3")

copy_paste_augmentation(dataset_source_path, segmentation_path, dataset_target_path, num_copies=2, overwrite=True)