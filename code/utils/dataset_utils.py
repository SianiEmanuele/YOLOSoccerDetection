import os
import cv2
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

def extract_ball_objects(image_folder, label_folder, segmented=False, ball_class=0):
    """
    Extracts small object crops (balls) from dataset.
    :param image_folder: Path to images
    :param label_folder: Path to YOLO labels
    :param ball_class: Class index of the ball
    :return: List of (cropped ball image, bounding box)
    """
    ball_objects = []

    for img_file in os.listdir(image_folder):
        if not img_file.endswith(".jpg"):
            continue

        img_path = os.path.join(image_folder, img_file)
        label_path = os.path.join(label_folder, img_file.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        if not segmented:

            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x, y, bw, bh = map(float, line.strip().split())

                    if int(cls) == ball_class:
                        # Convert YOLO format to pixel coordinates
                        x1 = int((x - bw / 2) * w)
                        y1 = int((y - bh / 2) * h)
                        x2 = int((x + bw / 2) * w)
                        y2 = int((y + bh / 2) * h)

                        ball_crop = img[y1:y2, x1:x2]  # Crop the ball
                        if ball_crop.size > 0:
                            ball_objects.append((ball_crop, (x, y, bw, bh)))

        if segmented:
            # Crea una maschera binaria vuota
            mask = np.zeros((h, w), dtype=np.uint8)

            with open(label_path, "r") as f:
                for line in f:
                    data = list(map(float, line.strip().split()))
                    cls = int(data[0])
                    coords = np.array(data[1:]).reshape(-1, 2)

                    # Converti coordinate normalizzate in pixel
                    coords[:, 0] *= w  # x
                    coords[:, 1] *= h  # y
                    points = coords.astype(np.int32)

                    # Disegna il poligono sulla maschera se la classe corrisponde
                    if cls == ball_class:
                        cv2.fillPoly(mask, [points], 255)

            # Trova i contorni dell'oggetto segmentato
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Crea una maschera trasparente per il contorno
                alpha = np.zeros_like(mask)
                cv2.drawContours(alpha, [contour], -1, 255, thickness=cv2.FILLED)

                # Aggiungi il canale alfa all'immagine originale
                bgr = cv2.bitwise_and(img, img, mask=alpha)
                rgba = cv2.merge([bgr, alpha])

                # Ritaglia l'area effettiva della maschera (bounding box)
                x, y, bw, bh = cv2.boundingRect(contour)
                cropped = rgba[y:y + bh, x:x + bw]

                if cropped.size > 0:
                    ball_objects.append((cropped, (x, y, bw, bh)))

    print(f"Extracted {len(ball_objects)} ball objects for augmentation.")
    return ball_objects

def copy_paste_augmentation(img, labels, objects, segmented=False, min_balls=3, max_balls=6):
    """
    Apply copy-paste augmentation by inserting extracted objects into a new image.
    :param img: Original image (numpy array)
    :param labels: Labels for the image (YOLO format)
    :param objects: List of extracted small objects (e.g., footballs)
    :param paste_prob: Probability of applying augmentation
    :return: Augmented image and updated labels
    """
    
    # if augment first save
    h, w, _ = img.shape
    new_labels = labels.copy()

    # avoid modifying original image and labels
    img = img.copy()
    new_labels = labels.copy()

    # Randomly select number of objects to paste
    num_objects = random.randint(min_balls, max_balls)
    objs_to_copy = random.sample(objects, num_objects)

    for obj in objs_to_copy:
        
        obj_img, bbox = obj
        bx, by, bw, bh = bbox

        # Get a random position in the image
        x_center = random.randint(int(0.25*w), int(0.66*w) - obj_img.shape[1])
        y_center = random.randint(int(0.25*h), int(0.66*h) - obj_img.shape[0])

        if not segmented:
            # Paste the object onto the image
            img[y_center:y_center+obj_img.shape[0], x_center:x_center+obj_img.shape[1]] = obj_img
        
        else:
            # Separazione dei canali BGR e Alfa
            bgr = obj_img[:, :, :3]
            alpha = obj_img[:, :, 3] / 255.0  # Normalizza alfa tra 0 e 1

            # Incollaggio dell'oggetto rispettando la trasparenza
            for c in range(3):  # Per ogni canale BGR
                img[y_center:y_center + bh, x_center:x_center + bw, c] = (
                    alpha * bgr[:, :, c] + 
                    (1 - alpha) * img[y_center:y_center + bh, x_center:x_center + bw, c]
                )

        # Convert new object position into YOLO format
        x_norm = (x_center + obj_img.shape[1] / 2) / w
        y_norm = (y_center + obj_img.shape[0] / 2) / h
        w_norm = obj_img.shape[1] / w
        h_norm = obj_img.shape[0] / h

        # Append new object label
        new_labels.append([0, x_norm, y_norm, w_norm, h_norm])

    return img, new_labels

def bbox_copy_paste(dataset_root_folder, dataset_destination_path, num_copies, segmented=False, overwrite=False):
    """
    Processes a dataset by:
    - Copying validation & test images/labels without augmentation
    - Augmenting training images with copy-paste augmentation
    - Maintaining YOLO folder structure

    :param root_folder: Path to the original dataset (must contain 'train', 'valid', 'test' folders)
    :param dataset_destination_path: Path to save processed dataset
    :param num_copies: Number of augmented copies to generate for each training image
    :param overwrite: Whether to overwrite existing image
    """

    # Define dataset subfolders
    dataset_splits = ["train", "valid", "test"]

    # Create destination folders
    for split in dataset_splits:
        os.makedirs(os.path.join(dataset_destination_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_destination_path, split, "labels"), exist_ok=True)

    # copy data.yaml file
    shutil.copy(os.path.join(dataset_root_folder, "data.yaml"), os.path.join(dataset_destination_path, "data.yaml"))
    

    for split in ["valid", "test"]:
        img_folder = os.path.join(dataset_root_folder, split, "images")
        label_folder = os.path.join(dataset_root_folder, split, "labels")

        for img_file in os.listdir(img_folder):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(img_folder, img_file)
                label_path = os.path.join(label_folder, img_file.replace(".jpg", ".txt"))

                # Copy image
                img_save_path = os.path.join(dataset_destination_path, split, "images", img_file)
                cv2.imwrite(img_save_path, cv2.imread(img_path))

                # Copy label
                label_save_path = os.path.join(dataset_destination_path, split, "labels", img_file.replace(".jpg", ".txt"))
                with open(label_path, "r") as f_src, open(label_save_path, "w") as f_dst:
                    for line in f_src:
                        values = line.strip().split()
                        class_id = int(float(values[0]))  # Ensure class ID is an integer
                        bbox_values = [f"{float(v)}" for v in values[1:]] 
                        f_dst.write(f"{class_id} " + " ".join(bbox_values) + "\n")


    train_img_folder = os.path.join(dataset_root_folder, "train", "images")
    train_label_folder = os.path.join(dataset_root_folder, "train", "labels")

    # Extract ball objects once
    if not segmented:
        ball_objects = extract_ball_objects(train_img_folder, train_label_folder)
    else:
        seg_img_folder = os.path.join(dataset_root_folder, "..", "segmented_dataset", "images")
        seg_label_folder = os.path.join(dataset_root_folder, "..", "segmented_dataset", "labels")
        ball_objects = extract_ball_objects(seg_img_folder, seg_label_folder, segmented)
    
    print(f"Extracted {len(ball_objects)} ball objects for augmentation.")

    for img_file in os.listdir(train_img_folder):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(train_img_folder, img_file)
            label_path = os.path.join(train_label_folder, img_file.replace(".jpg", ".txt"))

            img = cv2.imread(img_path)

            # Load labels
            labels = []
            with open(label_path, "r") as f:
                labels = [list(map(float, line.strip().split())) for line in f.readlines()]

            # Save original image & labels
        
            img_save_path = os.path.join(dataset_destination_path, "train", "images", img_file)
            label_save_path = os.path.join(dataset_destination_path, "train", "labels", img_file.replace(".jpg", ".txt"))

            
            if not overwrite:
                    cv2.imwrite(img_save_path, img)
                    with open(label_path, "r") as f_src, open(label_save_path, "w") as f_dst:
                        for line in f_src:
                            values = line.strip().split()
                            class_id = int(float(values[0]))  # Ensure class ID is an integer
                            bbox_values = [f"{float(v)}" for v in values[1:]]
                            f_dst.write(f"{class_id} " + " ".join(bbox_values) + "\n")

            # Apply augmentation 
            for i in range(num_copies):
                
                img_aug, labels_aug = copy_paste_augmentation(img, labels, ball_objects, segmented)
                aug_img_file = f"aug_{i}_{img_file}"
                aug_label_file = aug_img_file.replace(".jpg", ".txt")

                aug_img_save_path = os.path.join(dataset_destination_path, "train", "images", aug_img_file)
                aug_label_save_path = os.path.join(dataset_destination_path, "train", "labels", aug_label_file)

                cv2.imwrite(aug_img_save_path, img_aug)
                with open(aug_label_save_path, "w") as f:
                    for label in labels_aug:
                        class_id = int(label[0])  # Ensure integer class ID
                        bbox_values = [f"{float(v)}" for v in label[1:]]
                        f.write(f"{class_id} " + " ".join(bbox_values) + "\n")


            else: 
                cv2.imwrite(img_save_path, img)
                with open(label_path, "r") as f_src, open(label_save_path, "w") as f_dst:
                        for line in f_src:
                            values = line.strip().split()
                            class_id = int(float(values[0]))  # Ensure class ID is an integer
                            bbox_values = [f"{float(v)}" for v in values[1:]] 
                            f_dst.write(f"{class_id} " + " ".join(bbox_values) + "\n")

    print("Dataset processing completed successfully!")

if __name__ == "__main__":
    cwd = os.getcwd()
    dataset_root_folder = os.path.join(cwd, "dataset")
    print(dataset_root_folder)

    dataset_source_path = os.path.join(dataset_root_folder, "yolov9", "v0")
    dataset_destination_path = os.path.join(dataset_root_folder, "yolov9", "v3")

    bbox_copy_paste(dataset_source_path, dataset_destination_path, num_copies=2, segmented=True, overwrite=False)