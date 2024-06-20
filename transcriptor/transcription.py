import re
import os


class Transcription:
    def __init__(self, audio_file_path: str, transcriptions: list[str]):
        self.audio_file_path = audio_file_path
        self.filename = self.find_name(audio_file_path)
        self.transcriptions = self.group_by_speaker(transcriptions)

    def __repr__(self) -> str:
        return "\n".join([f"\033[93m{speaker}:\033[0m\n {text}" for speaker, text in self.transcriptions])

    def find_name(self, file_path) -> str:
        pattern = r'/(.+?)\.wav'
        match = re.search(pattern, file_path)
        if match:
            return match.group(1)
        return "transcript"

    def group_by_speaker(self, transcriptions: list[str]):
        speaker_transcriptions = []
        previous_speack = transcriptions[0][0]
        speaker_text = ""
        for speaker, text in transcriptions:
            if speaker == previous_speack:
                speaker_text += text
            else:
                speaker_transcriptions.append((previous_speack, speaker_text))
                speaker_text = text
                previous_speack = speaker
        speaker_transcriptions.append((previous_speack, speaker_text))
        return speaker_transcriptions

    def save(self, directory: str = "transcripts") -> None:
        if not self.transcriptions:
            raise ValueError("No transcriptions available to save.")

        if not os.path.exists(directory):
            os.makedirs(directory)

        saving_path = os.path.join(
            directory, f"{self.filename}_transcript.txt")
        with open(saving_path, 'w') as f:
            for (speaker, text) in self.transcriptions:
                if text:
                    f.write(f"{speaker}: {text}\n")
                    print(f"Transcription saved to {saving_path}")
