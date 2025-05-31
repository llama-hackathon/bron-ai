import subprocess
import cv2
from moviepy import *
import time
import base64
import os
from llama_api_client import LlamaAPIClient
from dotenv import load_dotenv
import json
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class Video:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.name = os.path.splitext(os.path.basename(self.filepath))[0]
        self.name_no_ext = os.path.splitext(os.path.basename(self.filepath))[0]

    def extract_frames(self, seconds_per_frame=2) -> dict:

        frame_dict = {}
        base_video_path, _ = os.path.splitext(self.filepath)

        video = cv2.VideoCapture(self.filepath)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int(fps * seconds_per_frame)
        curr_frame=0

        # Create frames directory if it doesn't exist
        frames_dir = "data/frames"
        os.makedirs(frames_dir, exist_ok=True)

        # Loop through the video and extract frames at specified sampling rate
        frame_count = 0
        while curr_frame < total_frames - 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            
            # Save frame as JPEG file
            frame_seconds = curr_frame / fps if fps else 0
            frame_filename = f"{self.name}_{frame_count:04d}_{frame_seconds:.2f}s.jpg"

            frame_filepath = os.path.join(
                frames_dir,
                f"{self.name}_{frame_count:04d}_{frame_seconds:.2f}s.jpg"
            )
            cv2.imwrite(frame_filepath, frame)
            frame_count += 1
            curr_frame += frames_to_skip

            frame_dict[frame_seconds] = {
                'data': base64.b64encode(buffer).decode("utf-8"),
                'source': frame_filename,
                'source_type': 'frame'
            }

        video.release()

        print(f"Extracted {len(frame_dict)} frames")
        print(f"Saved {frame_count} frames to {frames_dir}")
        return frame_dict

    def describe_frames(self, frames: dict):

        client = LlamaAPIClient(
            api_key=os.environ.get("LLAMA_API_KEY"),
        )
        for seconds, frame in frames.items():
            print(f"Describing frame at {seconds} seconds...")
            response = client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What does this image contain?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame['data']}"
                                },
                            },
                        ],
                    },
                ],
            )
            frame['annotation'] = response.completion_message.content.text

        return frames


    def extract_audio(self, audio_filepath: str):
        """
        Extract audio from the video file and save it as a WAV file.
        
        :param output_audio: Path to save the extracted audio file.
        """
        subprocess.run([
            "ffmpeg", "-i", self.filepath, "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100", "-ac", "1", audio_filepath
        ])


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    video_file = os.path.join(base_dir, 'data', 'video', 'game.mp4')
    vid = Video(video_file)

    # 1. Extract 1 frame per minute (to start). You can adjust the `seconds_per_frame` parameter to change the sampling rate
    frame_dict = vid.extract_frames(seconds_per_frame=60)

    # 2. Describe frames
    frames = vid.describe_frames(frame_dict)
