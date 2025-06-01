import os
import sys
import json
from typing import List, Dict, Union
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.talk2Video import Talk2Video

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

ANALYSIS_SUMMARY_PROMPT = """
    You are a keen eyed basketball referee, watching annotations of basketball play.

    I want you to summarize the annotations of the play in a 2-3 sentences, focusing on the most relevant information for foul calling.
    The annotations are provided in a time-ordered list.
    You do not need to give a frame by frame description, just the most relevant information of the sequence. 
    Just return the summary as a single string, without any additional text.

    ANNOTATIONS:
    ANNOTATION_DATA_TOKEN
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
        • Contact on the shooter's head/neck (automatic)  
        • Push from behind or on the side causing altered shot trajectory


        annotation data:

        ANNOTATION_DATA_TOKEN"""

class Referee:
    def __init__(self, video_filepath: str):
        self.talk_to_video = Talk2Video(video_filepath)

    def look_into_video(self, timestamp: int, boundry_seconds: int = 2):
        """
        Look into a video file and extract frames for annotation.
        """

        # focus on frames around the timestamp
        frames = self.talk_to_video.vid.cut_frames(max(timestamp-boundry_seconds, 0), timestamp + boundry_seconds, seconds_per_frame=0.2)

        keys = list(frames.keys())
        keys.sort(key=float)

        groups = [keys[i:i + 7] for i in range(0, len(keys), 6)]

        analyses = []
        for group in groups:
            print(f"Analyzing group: {group}")
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
            resp = self.talk_to_video.vid.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=messages,
            )
            analyses.append(resp.completion_message.content.text)

        # summarise the analyses
        summary_raw = self.talk_to_video.vid.client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": ANALYSIS_SUMMARY_PROMPT.replace("ANNOTATION_DATA_TOKEN", "\n".join(analyses)),
                        }
                    ]
                }
            ],
        )
        summary = summary_raw.completion_message.content.text
        print(f"Summary of the play: {summary}")
        return summary


    def fan_aligned_judgement(self, analysis: str):
        """
        Make a judgement using fan-aligned LLM.
        """
        print(f"Fan analysis: {analysis}")

        pass

    def make_judgement(self, analyses: Union[str,list]):
        """
        Make a judgement based on generic .
        """
        # print("\n--------------------------------\n".join(analyses))
        # print('--------------------------------')

        if isinstance(analyses, list):
            analyses = "\n".join(analyses)

        judgement = self.talk_to_video.vid.client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": FOUL_JUDGEMENT_PROMPT.replace("ANNOTATION_DATA_TOKEN", analyses),
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

        # return json_annotations, json_judgement, is_foul_present
        return is_foul_present
    
if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    video_file = os.path.join(base_dir, 'data', 'video', 'game_2_60.mp4')
    # video_file = os.path.join(base_dir, 'data', 'foulless_play.mp4')

    ref = Referee(video_file)

    res = []
    # for interesting_ts in [22, 152, 177, 217, 262, 337, 477, 582, 22, 152, 177, 217, 262, 272, 337, 362, 477, 582, 637, 647, 672, 767, 817, 927, 937, 1127, 1132, 1177, 1187]:
    for interesting_ts in [337]:
        summary = ref.look_into_video(interesting_ts, boundry_seconds=2)
        result = ref.make_judgement(summary)
        print(f"Foul is present: {result}")
        #res.append((interesting_ts, summary))

    # pipe separated csv
    # import csv
    # with open('res.csv', 'w') as f:
    #     writer = csv.writer(f, delimiter='|')
    #     writer.writerows(res)
