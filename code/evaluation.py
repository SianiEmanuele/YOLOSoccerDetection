from dotenv import load_dotenv
from utils import read_video, save_video
from trackers import Tracker
import os
import cv2
from ultralytics import YOLO
from models import SRYOLO
import matplotlib.pyplot as plt

def annotate_predictions(predictions, tracker):
        drawn_images = []
        test_images = []
        for pred in predictions:
            image = pred.orig_img
            test_images.append(image)
            drawn_frame = image.copy()
            bboxes = pred.boxes.data

            for bbox in bboxes:
                bbox = bbox.tolist()
                class_id = int(bbox[5])
                if class_id == 0:
                    color = (0, 255, 0)
                    drawn_frame = tracker.draw_triangle(drawn_frame, bbox[:4], color, bbox[4])
                    
                if class_id == 1:
                    color = (255, 0, 0)
                    drawn_frame = tracker.draw_ellipse(drawn_frame, bbox[:4], color, bbox[4])

                if class_id == 2:
                    color = (0, 0, 255)
                    drawn_frame = tracker.draw_ellipse(drawn_frame, bbox[:4], color, bbox[4])

                if class_id == 3:
                    color = (255, 255, 0)
                    drawn_frame = tracker.draw_ellipse(drawn_frame, bbox[:4], color, bbox[4])

            drawn_images.append(drawn_frame)
        return drawn_images, test_images

def detect_video(video_path, model_path, output_path):
    video_frames = read_video(video_path)
    tracker = Tracker(model_path=model_path)
    tracks = tracker.get_object_tracks(video_frames)
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, output_path)


def detect_images(images_path, model_path, is_sr=False, gan_weights=None):
    test_path = images_path
    tracker = Tracker(model_path=model_path)
    if not is_sr:
        predictions = YOLO(model_path).predict(test_path)
    else:
        predictions = SRYOLO(
            yolo_weights=model_path,
            upscale=4,
            gan_weights=gan_weights,
            dni_weight=0.5,
            tile=0,
            tile_pad=10
        ).predict(source=test_path)
    drawn_images, test_images = annotate_predictions(predictions, tracker)
    
    n_images = len(drawn_images)
    
    # Create figure with height proportional to number of images, width fixed
    plt.figure(figsize=(8, 6 * n_images))  # width=8, height=6 per image
    
    for i, img in enumerate(drawn_images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Arrange subplots in a single column
        plt.subplot(n_images, 1, i + 1)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Image {i}")
    
    plt.tight_layout()
    plt.show()


    

def main():
    cwd = os.getcwd()
    dataset_root_folder = os.path.join(cwd, "dataset", "yolov9", "v3")
    images_input_path = os.path.join(dataset_root_folder,"test", "images")
    model_path = r"src\training\yolo_football_analysis\yolo9c_dataset_v3_high_res2\weights\best.pt"
    model_name = "yolo_dataset_v3_high_res"
    output_path = r"output_images"  

    detect_images(images_input_path, model_path, output_path, model_name)
                


if __name__ == '__main__':
    main()