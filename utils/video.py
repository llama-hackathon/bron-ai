import subprocess


class Video:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def extract_audio(self, audio_filepath: str):
        """
        Extract audio from the video file and save it as a WAV file.
        
        :param output_audio: Path to save the extracted audio file.
        """
        subprocess.run([
            "ffmpeg", "-i", self.filepath, "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100", "-ac", "1", audio_filepath
        ])


if __name__ == "__main__":
    # Example usage
    video = Video("/Users/james/Documents/Projects/bron-ai/data/video/game.mp4")
    audio_output = "/Users/james/Documents/Projects/bron-ai/data/audio/game.wav"

    video.extract_audio(audio_output)
    print(f"audio file created at: {video.filepath}")