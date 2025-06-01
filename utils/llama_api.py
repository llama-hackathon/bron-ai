import os
from dotenv import load_dotenv
import math
from llama_api_client import RateLimitError  # Import the specific error
import time
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))


class LlamaAPI():
    def __init__(self):
        from llama_api_client import LlamaAPIClient  # Adjust import as needed
        self.client = LlamaAPIClient(
            api_key=os.environ.get("LLAMA_API_KEY"),
        )



    def ask(self, messages, model='Llama-4-Scout-17B-16E-Instruct-FP8', max_retries=3, retry_delay=15):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return response
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise
            except Exception as e:
                raise

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