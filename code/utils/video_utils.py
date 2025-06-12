import cv2

def read_video(video_path):
    """Read video file and return frames."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    """Save video file from frames."""
    fourcc = cv2.VideoWriter.fourcc(*"XVID") # output format
    out = cv2.VideoWriter(str(output_video_path), fourcc, 24.0, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()