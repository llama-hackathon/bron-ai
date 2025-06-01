import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.video import Video

class Referee:
    def __init__(self):
        pass

    def look_into_video(self, video_filepath: str):
        """
        Look into a video file and extract frames for annotation.
        """
        vid = Video(video_filepath)
        frame_dict = vid.extract_frames(seconds_per_frame=1)

