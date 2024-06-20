import re
import os


class Transcription:
    def __init__(self, audio_file_path: str, transcriptions: list[str]):
        self.audio_file_path = audio_file_path
        self.filename = self.find_name(audio_file_path)
        self.transcriptions = transcriptions

    def __repr__(self) -> str:
        return "\n".join([f"Speaker: {speaker}, Text: {text}" for speaker, text in self.transcriptions])

    def find_name(self, file_path) -> str:
        pattern = r'/(.+?)\.wav'
        match = re.search(pattern, file_path)
        if match:
            return match.group(1)
        return "transcript"

    def save(self, directory: str = "transcripts") -> None:
        if not self.transcriptions:
            raise ValueError("No transcriptions available to save.")

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, f"{self.filename}_transcript.txt"), 'w') as f:
            for (speaker, text) in self.transcriptions:
                if text:
                    f.write(f"{speaker}: {text}\n")
