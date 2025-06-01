import os
import json
import sys
import numpy as np
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llama_api import LlamaAPI


class Talk2Video:
    def __init__(self, annotations_path: str):
        self.annotations_path = annotations_path
        self.annotations = self.load_annotations()
        self.llama_api = LlamaAPI()

        # Create a smaller version of self.annotations: {timestamp: annotation}
        self.simple_annotations = {str(k): v.get("annotation", "") for k, v in self.annotations.items()}
    
    def load_annotations(self) -> dict:
        """Load annotations from a JSON file."""
        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")
        
        with open(self.annotations_path, 'r') as f:
            return dict(sorted(json.load(f).items(), key=lambda item: float(item[0])))
    

    def chunk_annotations(self, annotations:dict, token_limit: int = 128000) -> list:
        """
        Splits a dictionary into a list of dictionaries, each not exceeding the token_limit
        when serialized to JSON and tokenized (approximate, using 4 chars per token).
        """

        sub_dicts = []
        current_dict = {}
        current_tokens = 0

        for k, v in annotations.items():
            item_json = json.dumps({k: v})
            item_tokens = self.llama_api.estimate_tokens(item_json)
            if current_tokens + item_tokens > token_limit and current_dict:
                sub_dicts.append(current_dict)
                current_dict = {}
                current_tokens = 0
            current_dict[k] = v
            current_tokens += item_tokens

        if current_dict:
            sub_dicts.append(current_dict)
        return sub_dicts
    
    def summarize_annotations(self) -> str:
        """
        Use the annotations as context and ask llama_api to create a high-level summary of the video.
        """

        # Split simple_annotations into chunks that fit within the token limit
        annotation_chunks = self.chunk_annotations(annotations = self.simple_annotations, token_limit=120000)

        summaries = []
        for i, chunk in enumerate(annotation_chunks):
            print(f"Processing chunk with {i} / {len(annotation_chunks)} annotations...")
            context = json.dumps(chunk)
            messages = [
            {
                "role": "system",
                "content": (
                "You are an expert annotation summarizer.\n"
                "Your task is to take a series of timestamped annotations and create a sequential summary of what happened in the video.\n"
                "The annotations give you the timestamp in seconds, along with a summary of the content of the frame or audio."
                ),
            },
            {
                "role": "user",
                "content": context
            }
            ]
            response = self.llama_api.ask(messages, model='Llama-4-Maverick-17B-128E-Instruct-FP8')
            summaries.append(response.completion_message.content.text if hasattr(response, "completion_message") else str(response))

        aggregated_summary = "\n".join(summaries)

        # summarize the aggregated summary
        final_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert annotation summarizer.\n"
                    "Your task is to take a series of timestamped annotations and create a sequential summary of what happened in the video.\n"
                ),
            },
            {
                "role": "user",
                "content": aggregated_summary
            }
        ]
        response = self.llama_api.ask(final_messages, model='Llama-4-Maverick-17B-128E-Instruct-FP8')
        return aggregated_summary
        

    def look_for_event(self, event: str, window_length:int = 5, search_start:int=0, search_end=np.inf) -> list:
        """
        Compiles annotations within a specified window range.
        Checks if the event is present within the window range.
        Returns a list of timestamps where the event occurs.
        """

        timestamps = [float(ts) for ts in self.simple_annotations.keys()]
        timestamps = [ts for ts in timestamps if search_start <= ts < search_end]
        event_timestamps = []

        # Create windows by looping through the timestamps

        def search_window(args):
            start, window_length, event, simple_annotations = args
            end = start + window_length
            window_annotations = {
                k: v for k, v in simple_annotations.items() if start <= float(k) < end
            }
            if window_annotations:
                context = json.dumps(window_annotations)
                messages = [
                    {
                    "role": "system",
                    "content": (
                        "You are an expert event detector. The user will provide you with a set of video annotations from any domain.\n"
                        "Your task is to determine whether a specific event occurred within the video based solely on these annotations.\n"
                        "The event to check for is: " + event + "\n"
                        "Remember the annotations are sampled, so they may not cover every frame of the video.\n"
                        "Respond with only 'yes' if it looks the event is very likely to have occured within the video window, otherwise respond with only 'no'."
                    ),
                    },
                    {
                    "role": "user",
                    "content": context
                    }
                ]
            response = self.llama_api.ask(messages, model='Llama-4-Maverick-17B-128E-Instruct-FP8')
            result = response.completion_message.content.text.strip().lower() if hasattr(response, "completion_message") else str(response).strip().lower()

            if "yes" in result:
                return start + window_length // 2  # Add the midpoint of the window
            return None

        window_args = [
            (start, window_length, event, self.simple_annotations)
            for start in range(int(min(timestamps)), int(max(timestamps)) + 1, window_length)
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(search_window, window_args))

        event_timestamps.extend([r for r in results if r is not None])
        print(event_timestamps)




if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    annotations_file = os.path.join(base_dir, 'data', 'annotations', 'game_20_annotations.json')
    
    talk2video = Talk2Video(annotations_file)

    # summary = talk2video.summarize_annotations()
    # print("Video Summary:")
    # print(summary)

    talk2video.look_for_event("""
        • Illegal contact with the shooter's arms, wrist, or hand on the ball
        • Body-to-body displacement that affects balance or verticality
        • Defender invading the shooter's landing space (counter to rule 10-IV-f)
        • Contact on the head/neck or airborne shooter (automatic)
        • Push from behind or on the side causing altered shot trajectory
""", 
                              window_length=5,
                              search_start=60*10,
                              search_end=60*20)  # Search within the first 5 minutes