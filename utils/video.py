import subprocess
import cv2
from moviepy import *
import time
import base64
import os
from llama_api_client import LlamaAPIClient
from dotenv import load_dotenv
import json
import concurrent.futures
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

    def describe_frames(self, frames: dict, context: str = None, threads:int = 20) -> dict:

        client = LlamaAPIClient(
            api_key=os.environ.get("LLAMA_API_KEY"),
        )

        def describe_frame(args):
            seconds, frame = args
            print(f"Describing frame at {seconds} seconds...")
            response = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": f"Provide a couple sentences describing what is in this image. {context if context else ''} ",
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
            return seconds, response.completion_message.content.text

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            results = list(executor.map(describe_frame, frames.items()))

        for seconds, annotation in results:
            frames[seconds]['annotation'] = annotation

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

def parse_foul_info(v_path):
    prompt =     """
        You are a keen eyed basketball referee, watching a basketball play. 
    
        Analyze the provided play (represented by a sequence of basketball image frames) for detail relevant for foul calling, especially any contact. If there is contact, you sohuld
        describe it in detail - what body part was hit, what contact type was used, how severe the contact was. Since the frames comprise a single play, you should interpret the play in the
        context *all* the provided images. A single image might not provide enough information. Each frame is 50 milliseconds apart!

        Your job is not to call fouls, but to describe the action in detail for someone else to call fouls.
        
        We are only concerned with the ball handler/shooter and the primary defender.

        The overall context of the sequence is known to be a basketball game, so you can focus on the actions of the ball handler and the primary defender.

        Do not try to identify the players or teams, just refer to them as players on offense/defense, etc.

        You do not need to give a frame by frame description, just the most relevant information of the sequence. 

    """

    client = LlamaAPIClient(
        api_key=os.environ.get("LLAMA_API_KEY"),
    )

    v = Video(v_path)
    frames = v.extract_frames(seconds_per_frame=.1)

    keys = list(frames.keys())
    keys.sort(key=float)

    print(keys)
    groups = [keys[i:i + 7] for i in range(0, len(keys), 7)]
    print(groups)
    analyses = []
    for group in groups:
        messages = []
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ]
        })
        for k in group:
            frame = frames[k]
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame['data']}"
                        },
                    },
                ]
            })
        resp = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=messages,
        )
        analyses.append(resp.completion_message.content.text)

    print("\n|||\n".join(analyses))

    judgement_prompt = '''
        You are a seasoned basketball referee with expert knowledge of NBA rules
        for defensive fouls on a shooter.

        **Your job**

        Given a time-ordered list of annotations of a single basketball play, 
        decide whether the **primary defender commits a defensive
        shooting foul** during the shot attempt described.

        Despite being so information dense, these annotations still correspond to a single basketball play taking place over a couple of seconds! it is all one moment to be taken in at once, and the 
        amount of information is provided in order to give you enough minute information to make your decision: to call foul or not foul.

         Look for the following foul indicators _during the shooting motion_  

        • Illegal contact with the shooter's arms, wrist, or hand on the ball  
        • Body-to-body displacement that affects balance or verticality  
        • Defender invading the shooter's landing space (counter to rule 10-IV-f)  
        • Contact on the head/neck or airborne shooter (automatic)  
        • Push from behind or on the side causing altered shot trajectory


        annotation data:

        ANNOTATION_DATA_TOKEN
   '''

    print('--------------------------------')
    judgement = client.chat.completions.create(
        model="Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": judgement_prompt.replace("ANNOTATION_DATA_TOKEN", "\n".join(analyses)),
                    }
                ]
            }
        ],
    )
    print(judgement.completion_message.content.text)

if __name__ == "__main__":
    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    video_file = os.path.join(base_dir, 'data', 'nba_2016_finals_6_clip.mp4')
    # video_file = os.path.join(base_dir, 'data', 'foulless_play.mp4')

    parse_foul_info(video_file)

    # vid = Video(video_file)

    # # 1. Extract 1 frame per minute (to start). You can adjust the `seconds_per_frame` parameter to change the sampling rate
    # frame_dict = vid.extract_frames(seconds_per_frame=60)

    # # 2. Describe frames
    # frames = vid.describe_frames(frame_dict)