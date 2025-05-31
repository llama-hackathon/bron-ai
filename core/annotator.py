import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video import Video
from utils.audio import Audio
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class Annotator:
    def __init__(self, video_filepath):
        """Orchestrates the annotation process of both audio and video and produces a single annoation json file."""
        self.video_filepath = video_filepath
        self.vid = Video(self.video_filepath)

    def annotate_video(self):
        """Annotates a video file."""

        # 1. Extract 1 frame per minute (to start). You can adjust the `seconds_per_frame` parameter to change the sampling rate
        frame_dict = self.vid.extract_frames(seconds_per_frame=1)

        # 2. Describe frames
        frames = self.vid.describe_frames(frame_dict,
                                    context= ("This is a frame of tv footage of a basketball game, you don't need to mention that in your response. "
                                    "If the frame is of the basketball game in play and you can see the ball, only focus on the ball handler, what he is doing and how he is being defended."
                                    "Be specific with details a referee would be interested in. "
                                    "Assume that frames a second ago (and beyond) have already been annotated, so you can focus on the subject at hand. No need to 'start from scratch' in describing the game."
                                    "Ignore descriptions of the setting, stadium, crowd or the scoreboard on screen"),
                                    threads=10)
        self.save_annotations(frames, os.path.join("data", "annotations", f"{self.vid.name_no_ext}_annotations_vid.json"))


    def annotate_audio(self, video_filepath):

        # 1. Extract audio
        base_dir = os.path.dirname(os.path.abspath(video_filepath))
        audio_output = os.path.join(base_dir, '..', 'audio', f'{self.vid.name_no_ext}.wav')
        self.vid.extract_audio(audio_output)

        # 2. Transcribe audio
        audio = Audio(audio_output)
        transcription = audio.transcribe()
        transcription_segments = transcription.get('segments', [])
        frames = {}

        for segment in transcription_segments:
            start = segment['start']
            end = segment['end']
            middle = (start + end) / 2
            # add annotation to frames dict
            frames[middle] = {
                'source': f'{self.vid.name_no_ext}.wav',
                'source_type': 'audio',
                'annotation': segment.get('text', ''),
                'start': start,
                'end': end
            }

        # Save to a JSON file
        self.save_annotations(frames, os.path.join("data", "annotations", f"{self.vid.name_no_ext}_annotations_audio.json"))


    def compile_annotaions(self, video_annotations, audio_annotations):
        """Compiles video and audio annotations into a single dictionary."""
        compiled_annotations = {}
        for timestamp, annotation in video_annotations.items():
            # Exclude 'data' key from annotation if present
            annotation_no_data = {k: v for k, v in annotation.items() if k != 'data'}
            compiled_annotations[timestamp] = annotation_no_data

        for timestamp, annotation in audio_annotations.items():
            if timestamp not in compiled_annotations:
                compiled_annotations[timestamp] = annotation
            else:
                # If both video and audio annotations exist at the same timestamp, merge them
                compiled_annotations[timestamp].update(annotation)

        self.save_annotations(compiled_annotations, os.path.join("data", "annotations", f"{self.vid.name_no_ext}_annotations.json"))

    def save_annotations(self, annotations, output_path):
        """Saves the annotations to a JSON file."""
        with open(output_path, "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"Frame metadata saved to {output_path}")



if __name__ == "__main__":  
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    video_file_name = 'game_20.mp4'
    video_file = os.path.join(base_dir, 'data', 'video', video_file_name)

    annotator = Annotator(video_file)
    annotator.annotate_video()
    # annotator.annotate_audio(video_file)

    annotator.compile_annotaions(
        video_annotations=json.load(open(os.path.join("data", "annotations", f"{annotator.vid.name_no_ext}_annotations_vid.json"), 'r')),
        audio_annotations=json.load(open(os.path.join("data", "annotations", f"{annotator.vid.name_no_ext}_annotations_audio.json"), 'r'))
    )