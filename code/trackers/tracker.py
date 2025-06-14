import os
import pickle
import supervision as sv
import cv2
import numpy as np
from ultralytics import YOLO

from utils import get_bbox_width, get_center_of_bbox

class Tracker:
    """
        A class used to track objects in video frames using the YOLO model and ByteTrack.

        Attributes
        ----------
        model : YOLO
            The YOLO model used for object detection.
        tracker : sv.ByteTrack
            The ByteTrack tracker used for tracking detected objects.
        dev_mode : bool
            A flag indicating if the tracker is in development mode. Meaning that the tracker state is saved and loaded from a file.
        save_path : str
            The path where the tracker state is saved.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.dev_mode = os.getenv("DEV") == "True"
        self.save_path = os.getenv("TRACKER_SAVE_PATH")

    def draw_ellipse(self, frame, bbox, color, conf=None):
        """
        Draw an ellipse around a bounding box in a frame.
        :param frame: The frame to draw the ellipse on.
        :param bbox: The bounding box to draw the ellipse around.
        :param color: The color of the ellipse.
        :param conf: Confidence score of the detection (optional).
        :return: The frame with the ellipse drawn on it.
        """
        y2 = int(bbox[3])

        x_center, y_center = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )


        if conf is not None:
            # take only the first 2 decimal digits
            conf = int(conf * 100) / 100
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15
            cv2.rectangle(
                frame,
                (int(x1_rect),int(y1_rect)),
                (int(x2_rect),int(y2_rect)),
                color,
                thickness=cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if conf > 99:
                x1_text -=10

            cv2.putText(
                frame,
                f"{conf}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                thickness=2
            )
        return frame

    def draw_triangle(self, frame, bbox, color, conf=None):
        """
        Draw a triangular indicator on top of the ball.
        :param frame: The frame to draw the triangle on.
        :param bbox: The bounding box of the ball.
        :param color: The color of the triangle.
        :param conf: Confidence score of the detection (optional).
        :return: The frame with the triangle drawn on it.
        """
        y = int(bbox[1]) # y1 for triangle on top of the ball
        x_center, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([[x_center,y], [x_center-10,y-20], [x_center+10,y-20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) # border
        
        if conf is not None:
            conf = int(conf * 100) / 100

            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (int(bbox[1]) - rectangle_height // 2) + 30
            y2_rect = (int(bbox[1]) + rectangle_height // 2) + 30
            cv2.rectangle(
                frame,
                (int(x1_rect),int(y1_rect)),
                (int(x2_rect),int(y2_rect)),
                color,
                thickness=cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if conf > 99:
                x1_text -=10

            cv2.putText(
                frame,
                f"{conf}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                thickness=2
            )

        return frame

    def detect_frames(self, frames):
        """
        Detect objects in frames
        :param frames: list of frames
        :return: list of detections
        """
        batch_size = 20
        detections = []
        # loop through frames in batches to avoid memory issues
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames):
        """
        Get object tracks from a list of frames.

        This method detects objects in the provided frames and tracks them over time.
        It categorizes the detected objects into players, ball, referees.
        N.B. The goalkeepers are considered as players.

        :param frames: List of frames to process.
        :type frames: list
        :return: Dictionary containing tracks for players, ball, referees, and goalkeepers.
        :rtype: dict

        Example:
            "tracks":{
                "players": [
                {0: {"bbox": [x,y,x,y]}, 1: {"bbox": [x,y,x,y]}}, # frame 0
                {21: {"bbox": [x,y,x,y]}, 10: {"bbox": [x,y,x,y]}}, # frame 2
                ],
                "referees": [
                {12: {"bbox": [x,y,x,y]}, 13: {"bbox": [x,y,x,y]}}, # frame 0
                {12: {"bbox": [x,y,x,y]}, 13: {"bbox": [x,y,x,y]}}, # frame 2
                ],
                ...
            }
        """

        if self.dev_mode and os.path.exists(self.save_path):
            with open(self.save_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "ball": [],
            "referees": [],
            #"goalkeepers": []
        }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_inv_map = {v:k for k,v in class_names.items()} # switch key and value, from "0:player" to "player:0"

            # convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # TO BE DISCUSSED: do we need to convert the goalkeepers to players?
            # convert goalkeepers to players
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = class_inv_map["player"]

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})
            #tracks["goalkeepers"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist() # the first element is the bbox
                class_id = frame_detection[3] # the fourth element is the class id
                track_id = frame_detection[4]

                if class_id == class_inv_map["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if class_id == class_inv_map["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                # if class_id == class_inv_map["goalkeeper"]:
                #     tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}

            # since there is only one ball, we can hardcode the id to 1
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_inv_map["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox} # only one ball so id is 1

        if self.dev_mode:
            with open(self.save_path, "wb") as f:
                pickle.dump(tracks, f)
        return tracks

    def draw_annotations(self, video_frames, tracks):
        """
        Draw annotations on video frames. Calls all other draw methods.
        :param video_frames: List of video frames
        :param tracks: Dictionary containing tracks for players, ball, referees.
        :return: List of video frames with annotations drawn on them.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            players_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referees_dict = tracks["referees"][frame_num]

            # draw players

            for track_id, player in players_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0,0,255), track_id) # red

            for _, referee in referees_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255)) # yellow

            # Draw ball indicator
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))


            output_video_frames.append(frame)



        return output_video_frames