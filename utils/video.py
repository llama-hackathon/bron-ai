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


DETAIL_ANNOTATOR_PROMPT = """
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

FOUL_JUDGEMENT_PROMPT = """ 
        You are a seasoned basketball referee with expert knowledge of NBA rules
        for defensive fouls on a shooter.

        **Your job**

        Given a time-ordered list of annotations of a single basketball play, 
        decide whether the **primary defender commits a defensive
        shooting foul** during the shot attempt described.

        Despite being so information dense, these annotations still correspond to a single basketball play taking place over a couple of seconds! it is all one moment to be taken in at once, and the 
        amount of information is provided in order to give you enough minute information to make your decision: to call foul or not foul.

        <IMPORTANT>
        You can return free form text, but ensure you surround your final answer with backticks. either ```FOUL``` or ```CLEAN```
        </IMPORTANT>
        
         Look for the following foul indicators _during the shooting motion:

        • Illegal contact with the shooter's arms, wrist, or hand on the ball  
        • Body-to-body displacement that affects balance or verticality  
        • Defender invading the shooter's landing space (counter to rule 10-IV-f)  
        • Contact on the head/neck or airborne shooter (automatic)  
        • Push from behind or on the side causing altered shot trajectory


        annotation data:

        ANNOTATION_DATA_TOKEN"""

class Video:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.name = os.path.splitext(os.path.basename(self.filepath))[0]
        self.name_no_ext = os.path.splitext(os.path.basename(self.filepath))[0]
        self.client = LlamaAPIClient(
            api_key=os.environ.get("LLAMA_API_KEY"),
        )


    # def cut_video(self, start_time, end_time):



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
        self.frames = frame_dict
        return frame_dict

    def cut_frames(self, start_timestamp: float, end_timestamp: float, seconds_per_frame: float = 1.) -> dict:
        """
        Extract frames from video between start_timestamp and end_timestamp.
        
        :param start_timestamp: Start time in seconds
        :param end_timestamp: End time in seconds  
        :param seconds_per_frame: Time interval between extracted frames in seconds
        :return: Dictionary of extracted frames with timestamps as keys
        """
        frame_dict = {}
        
        video = cv2.VideoCapture(self.filepath)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0:
            video.release()
            raise ValueError("Could not determine video FPS")
            
        # Convert timestamps to frame numbers
        start_frame = int(start_timestamp * fps)
        end_frame = int(end_timestamp * fps)
        frames_to_skip = int(fps * seconds_per_frame)
        
        # Clamp frame numbers to valid range
        start_frame = max(0, start_frame)
        end_frame = min(total_frames - 1, end_frame)
        
        # Create frames directory if it doesn't exist
        frames_dir = "data/frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        curr_frame = start_frame
        frame_count = 0
        
        while curr_frame <= end_frame:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            if not success:
                break
                
            _, buffer = cv2.imencode(".jpg", frame)
            
            # Calculate actual timestamp for this frame
            frame_seconds = curr_frame / fps
            frame_filename = f"{self.name}_cut_{frame_count:04d}_{frame_seconds:.2f}s.jpg"
            
            frame_filepath = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_filepath, frame)
            
            frame_dict[frame_seconds] = {
                'data': base64.b64encode(buffer).decode("utf-8"),
                'source': frame_filename,
                'source_type': 'frame'
            }
            
            frame_count += 1
            curr_frame += frames_to_skip
            
        video.release()
        
        print(f"Extracted {len(frame_dict)} frames between {start_timestamp}s and {end_timestamp}s")
        print(f"Saved {frame_count} frames to {frames_dir}")
        
        return frame_dict

    def describe_frames(self, frames: dict, context: str = None, threads:int = 20) -> dict:


        def describe_frame(args):
            seconds, frame = args
            try:
                print(f"Describing frame at {seconds} seconds...")
                response = self.client.chat.completions.create(
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
            except Exception as e:
                print(f"Error describing frame at {seconds} seconds: {e}")
                return seconds, f"Error: {e}"

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

    def parse_foul_info(self, time_stamp, boundry_seconds=2):
        # client = LlamaAPIClient(
        #     api_key=os.environ.get("LLAMA_API_KEY"),
        # )

        # v = Video(v_path)
        # frames = v.extract_frames(seconds_per_frame=.1)
        # upperb, lowerb = time_stamp - boundry_seconds, time_stamp + boundry_seconds
        # keys = [k for k in self.frames.keys() if k >= lowerb and k <= upperb]
        
        # keys = list(frames.keys())
        # keys.sort(key=float)

        # groups = [keys[i:i + 7] for i in range(0, len(keys), 6)]

        frames = self.cut_frames(max(time_stamp-boundry_seconds, 0), time_stamp + boundry_seconds, seconds_per_frame=0.2)

        keys = list(frames.keys())
        keys.sort(key=float)

        groups = [keys[i:i + 7] for i in range(0, len(keys), 6)]



        # print(keys)
        # print(groups)
        analyses = []
        for group in groups:
            messages = []
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": DETAIL_ANNOTATOR_PROMPT,
                    },
                ]
            })
            for k in group:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frames[k]['data']}"
                            },
                        },
                    ]
                })
            resp = self.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=messages,
            )
            analyses.append(resp.completion_message.content.text)

        print("\n--------------------------------\n".join(analyses))
        print('--------------------------------')


        judgement = self.client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": FOUL_JUDGEMENT_PROMPT.replace("ANNOTATION_DATA_TOKEN", "\n".join(analyses)),
                        }
                    ]
                }
            ],
        )

        judge_text = judgement.completion_message.content.text
        print(judge_text)

        
        if "```FOUL```" in judge_text:
            is_foul_present =  True
        elif "```CLEAN```" in judge_text:
            is_foul_present = False
        else:
            print(f"Invalid response!!!!!: {judge_text}")
            is_foul_present = False

        clumped_annotations = "\n----\n".join(analyses)
        json_annotations = json.dumps(clumped_annotations)
        json_judgement = json.dumps(judge_text)

        return json_annotations, json_judgement, is_foul_present
    

if __name__ == "__main__":
    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    video_file = os.path.join(base_dir, 'data', 'nba_2016_finals_6.mp4')
    # video_file = os.path.join(base_dir, 'data', 'foulless_play.mp4')



    # parse_foul_info(video_file)

    vid = Video(video_file)
    # print(vid.parse_foul_info(60))


    res = []
    # for interesting_ts in [60, 120, 180]:
    for interesting_ts in [22, 152, 177, 217, 262, 337, 477, 582, 22, 152, 177, 217, 262, 272, 337, 362, 477, 582, 637, 647, 672, 767, 817, 927, 937, 1127, 1132, 1177, 1187]:
        json_escaped_annotations, judgement_text, is_foul_present = vid.parse_foul_info(interesting_ts)
        res.append((interesting_ts, json_escaped_annotations, judgement_text, is_foul_present))

    # pipe separated csv
    import csv
    with open('res.csv', 'w') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(res)




    # vid.cut_frames(60, 62)


    # # 1. Extract 1 frame per minute (to start). You can adjust the `seconds_per_frame` parameter to change the sampling rate
    # frame_dict = vid.extract_frames(seconds_per_frame=60)

    # # 2. Describe frames
    # frames = vid.describe_frames(frame_dict)