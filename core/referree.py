import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.video import parse_foul_info

class Referee:
    def __init__(self):
        pass

    def look_into_video(self, video_filepath: str):
        """
        Look into a video file and extract frames for annotation.
        """

        # 
        vid = parse_foul_info(video_filepath)


