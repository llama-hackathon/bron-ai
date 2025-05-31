import whisperx
import gc

class Audio:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_whisper_model(self, device: str = "cpu", compute_type: str = "float32"):
        # 1. Load and save original whisper model
        model_dir = "../models/"
        self.model = whisperx.load_model("turbo", device, compute_type=compute_type, download_root=model_dir)

    def transcribe(self, device: str = "cpu", batch_size: int = 6, compute_type: str = "float32"):

        self.load_whisper_model(device, compute_type)

        # 2. Transcribe audio
        audio = whisperx.load_audio(self.filepath)
        whisper_raw = self.model.transcribe(audio, batch_size=batch_size)

        # 3. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=whisper_raw["language"], device=device)
        whisper_aligned = whisperx.align(whisper_raw["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        print(whisper_aligned["segments"]) # after alignment



    
if __name__ == "__main__":
    # Example usage
    audio = Audio("/Users/james/Documents/Projects/bron-ai/data/audio/game.wav")
    audio.transcribe()
    