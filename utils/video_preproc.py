import cv2
from moviepy import *
import time
import base64
import os
import requests

def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
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
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        
        # Save frame as JPEG file
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        
        curr_frame += frames_to_skip
    video.release()

    # Extract audio from video
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Saved {frame_count} frames to {frames_dir}")
    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path


# Extract 1 frame per minute (to start). You can adjust the `seconds_per_frame` parameter to change the sampling rate
video_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'video', 'game.mp4')
base64Frames, audio_path = process_video(video_file, seconds_per_frame=60)



from llama_api_client import LlamaAPIClient
from dotenv import load_dotenv

# Load environment variables from .env file in the parent folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

client = LlamaAPIClient(
    api_key=os.environ.get("LLAMA_API_KEY"),
)

for frame in base64Frames:
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
                            "url": f"data:image/jpeg;base64,{frame}"
                        },
                    },
                ],
            },
        ],
    )
    print(response.completion_message.content.text)
