import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import math
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))


class LlamaAPI():
    def __init__(self):
        from llama_api_client import LlamaAPIClient  # Adjust import as needed
        self.client = LlamaAPIClient(
            api_key=os.environ.get("LLAMA_API_KEY"),
        )

    def ask(self, messages, model='Llama-4-Scout-17B-16E-Instruct-FP8'):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        print(response)
        return response

    @staticmethod
    def estimate_tokens(s):
        # Approximate: 1 token ≈ 4 characters (for English text)
        return math.ceil(len(s) / 4)

# Example usage:
# llama = LlamaAPI()
# result = llama.query([
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "What does this image contain?"},
#             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame['data']}"}},
#         ],
#     }
# ])