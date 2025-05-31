import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.llama_api import LlamaAPI


class Talk2Video:
    def __init__(self, annotations_path: str):
        self.annotations_path = annotations_path
        self.annotations = self.load_annotations()
        self.llama_api = LlamaAPI()

    
    def load_annotations(self) -> dict:
        """Load annotations from a JSON file."""
        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")
        
        with open(self.annotations_path, 'r') as f:
            return  dict(sorted(json.load(f).items(), key=lambda item: float(item[0])))
    

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
        # Create a smaller version of self.annotations: {timestamp: annotation}
        if isinstance(self.annotations, dict):
            self.simple_annotations = {str(k): v.get("annotation", "") for k, v in self.annotations.items()}
        else:
            self.simple_annotations = {}

        # Split simple_annotations into chunks that fit within the token limit
        annotation_chunks = self.chunk_annotations(annotations = self.simple_annotations, token_limit=120000)

        summaries = []
        for chunk in annotation_chunks:
            context = json.dumps(chunk)
            messages = [
            {
                "role": "system",
                "content": (
                "You are an expert video summarizer. Your task is to create a high-level sequential summary of a video based on its annotations.\n"
                "The annotations give you the timestamp in seconds, along with a summary of the content of the frame or audio."
                ),
            },
            {
                "role": "user",
                "content": context
            }
            ]
            response = self.llama_api.ask(messages, model='Llama-4-Scout-17B-16E-Instruct-FP8')
            summaries.append(response.completion_message.content.text if hasattr(response, "completion_message") else str(response))

        aggregated_summary = "\n".join(summaries)
        return aggregated_summary
        

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    annotations_file = os.path.join(base_dir, 'data', 'annotations', 'game_20_annotations.json')
    
    talk2video = Talk2Video(annotations_file)
    summary = talk2video.summarize_annotations()
    
    print("Video Summary:")
    print(summary)