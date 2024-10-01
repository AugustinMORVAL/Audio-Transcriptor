import os
from itertools import cycle
from termcolor import colored

class Transcription:
    """
    A class for storing and saving transcriptions.

    Attributes:
    -----------
    audio_file_path : str
        The path to the audio file that was transcribed.
    filename : str
        The name of the audio file, without the extension.
    transcriptions : list[str]
        A list of tuples containing the speaker's label and their corresponding transcription, grouped by speaker.
    speaker_names : dict
        A dictionary mapping speaker labels to their assigned names.
    segments : list
        A list of segments from diarization.

    """

    def __init__(self, audio_file_path: str, transcriptions: list[str], segments: list[str]):
        self.audio_file_path = audio_file_path
        self.filename = os.path.splitext(os.path.basename(audio_file_path))[0]
        self.transcriptions = self.group_by_speaker(transcriptions)
        self.speaker_names = {}
        self.segments = segments
        self.colors = cycle(['red', 'green', 'blue', 'magenta', 'cyan', 'yellow'])

    def __repr__(self) -> str:
        result = []
        for speaker, text in self.transcriptions:
            speaker_name = self.speaker_names.get(speaker, speaker)
            result.append(f"{speaker_name}:\n{text}")
        return "\n\n".join(result)

    def group_by_speaker(self, transcriptions: list[str]) -> list[str]:
        """
        Groups transcriptions by speaker.

        Parameters
        ----------
        transcriptions : list[str]
            A list of tuples containing the speaker's label and their corresponding transcription.

        Returns
        -------
        list[str]
            A list of tuples containing the speaker's label and their corresponding transcription, grouped by speaker.
        """
        speaker_transcriptions = []
        previous_speaker = transcriptions[0][0]
        speaker_text = ""
        for speaker, text in transcriptions:
            if speaker == previous_speaker:
                speaker_text += text
            else:
                speaker_transcriptions.append((previous_speaker, speaker_text))
                speaker_text = text
                previous_speaker = speaker
        speaker_transcriptions.append((previous_speaker, speaker_text))
        return speaker_transcriptions

    def save(self, directory: str = "transcripts") -> None:
        """
        Saves the transcription to a text file.

        Parameters
        ----------
        directory : str, optional
            The directory to save the transcription to. Defaults to "transcripts".
        """
        if not self.transcriptions:
            raise ValueError("No transcriptions available to save.")
        
        os.makedirs(directory, exist_ok=True)
        saving_path = os.path.join(directory, f"{self.filename}_transcript.txt")
        
        with open(saving_path, 'w', encoding='utf-8') as f:
            for speaker, text in self.transcriptions:
                if text:
                    speaker_name = self.speaker_names.get(speaker, speaker)
                    f.write(f"{speaker_name}: {text}\n")
        
        print(f"Transcription saved to {saving_path}")

    def get_name_speakers(self) -> None:
        """
        Interactively assigns names to speakers in the transcriptions and retrieves the name of the speaker.
        Provides a preview of one sentence for each speaker to help recognize who is speaking.
        """
        for speaker, full_text in self.transcriptions:
            if speaker in self.speaker_names:
                continue
            
            preview = full_text.split('.')[0] + '.'
            print(f"\nCurrent speaker: {speaker}")
            print(f"Preview: {preview}")
            
            new_name = input(f"Enter a name for {speaker} (or press Enter to skip): ").strip()
            if new_name:
                self.speaker_names[speaker] = new_name
                print(f"Speaker {speaker} renamed to {new_name}")
            else:
                print(f"Skipped renaming {speaker}")
        
        print("\nSpeaker naming completed.")
        print(f"Updated speaker names: {self.speaker_names}")
