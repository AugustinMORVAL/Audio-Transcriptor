import os

class Transcription:
    """
    A class for storing and saving transcriptions.

    Parameters
    ----------
    audio_file_path : str
        The path to the audio file that was transcribed.
    transcriptions : list[str]
        A list of tuples containing the speaker's label and their corresponding transcription.

    Attributes
    ----------
    audio_file_path : str
        The path to the audio file that was transcribed.
    filename : str
        The name of the audio file, without the extension.
    transcriptions : list[str]
        A list of tuples containing the speaker's label and their corresponding transcription, grouped by speaker.

    Usage
    -----
    >>> transcription = Transcription("/path/to/audio.wav", [("Speaker 1", "Transcription 1"), ("Speaker 1", "Transcription 2"), ("Speaker 2", "Transcription 3")])
    >>> print(transcription)
    >>> transcription.save("/path/to/transcripts")
    """

    def __init__(self, audio_file_path: str, transcriptions: list[str]):
        self.audio_file_path = audio_file_path
        self.filename = os.path.splitext(os.path.basename(audio_file_path))[0]
        self.transcriptions = self.group_by_speaker(transcriptions)

    def __repr__(self) -> str:
        return "\n".join([f"\033[93m{speaker}:\033[0m\n {text}" for speaker, text in self.transcriptions])

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

        Raises
        ------
        ValueError
            If there are no transcriptions available to save.
        """
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
