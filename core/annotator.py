import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video import Video
from utils.audio import Audio
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class Annotator:
    def __init__(self):
        """Orchestrates the annotation process of both audio and video and produces a single annoation json file."""



    def annotate_video(self, video_filepath):
        """Annotates a video file."""

        vid = Video(video_filepath)

        # 1. Extract 1 frame per minute (to start). You can adjust the `seconds_per_frame` parameter to change the sampling rate
        frame_dict = vid.extract_frames(seconds_per_frame=60)

        # 2. Describe frames
        frames = vid.describe_frames(frame_dict)

        # 3. Extract audio
        base_dir = os.path.dirname(os.path.abspath(video_filepath))
        audio_output = os.path.join(base_dir, '..', 'audio', f'{vid.name_no_ext}.wav')
        vid.extract_audio(audio_output)

        # 4. Transcribe audio
        audio = Audio(audio_output)
        transcription = audio.transcribe()
        transcription_segments = transcription.get('segments', [])

        for segment in transcription_segments:
            start = segment['start']
            end = segment['end']
            middle = (start + end) / 2
            frames[middle] = {
                'source': f'{vid.name_no_ext}.wav',
                'source_type': 'audio',
                'annotation': segment.get('text', ''),
                'start': start,
                'end': end
            }


        # Prepare a dict without the 'frame_data' (or 'data') key for each frame
        frames_to_save = {}
        for seconds, frame in frames.items():
            frame_copy = {k: v for k, v in frame.items() if k not in ('frame_data', 'data')}
            frames_to_save[seconds] = frame_copy

        # Save to a JSON file
        output_path = os.path.join("data", "annotations", f"{vid.name_no_ext}_annotations.json")
        with open(output_path, "w") as f:
            json.dump(frames_to_save, f, indent=2)
        print(f"Frame metadata saved to {output_path}")



if __name__ == "__main__":  
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    video_file_name = 'game.mp4'
    video_file = os.path.join(base_dir, 'data', 'video', video_file_name)

    annotator = Annotator()  # Replace `model=None` with your actual model if needed
    annotator.annotate_video(video_file)