from dotenv import load_dotenv
from utils import read_video, save_video
from trackers import Tracker
import os
import cv2
from ultralytics import YOLO
from models import SRYOLO

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

def detect_images(images_path, model_path, output_path, model_name):
    test_path = images_path
    tracker = Tracker(model_path=model_path)

    predictions = YOLO(model_path).predict(test_path)
    # predictions = SRYOLO(
    #     yolo_weights=model_path,
    #     scale=4,
    #     model_path=r'src\models\esrgan\experiments\finetune_Realesr-general-x4v3_2\models\net_g_latest.pth',
    #     dni_weight=0.5,
    #     tile=0,
    #     tile_pad=10,
    #     pre_pad=0,
    #     max_size=1280
    # ).predict(source=test_path)
    drawn_images, test_images = annotate_predictions(predictions, tracker)
    
    # Save the drawn images into a folder
    output_folder = os.path.join(output_path, model_name)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "predicted"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "input"), exist_ok=True)

    for i, (test_image, drawn_image) in enumerate(zip(test_images, drawn_images)):
        # Save the drawn image
        output_path = os.path.join(output_folder, "predicted", f"predicted_{i}.jpg")
        cv2.imwrite(output_path, drawn_image)
        # Save the original image
        input_path = os.path.join(output_folder, "input", f"input_{i}.jpg")
        cv2.imwrite(input_path, test_image)

    

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