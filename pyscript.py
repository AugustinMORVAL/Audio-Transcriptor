import os
from dotenv import load_dotenv
import whisper
from pyannote.audio import Audio, Pipeline
from pyannote.core import Segment
import numpy as np
import librosa
import tqdm
from time import sleep
import re

# Load environment variables from .env file
load_dotenv()

class Transcript:
    """
    A class for transcribing audio files.

    Parameters
    ----------
    audio_file : str
        The path to the audio file to be transcribed.

    Attributes
    ----------
    HF_TOKEN : str
        The Hugging Face token for accessing the PyAnnote speaker diarization pipeline.
    transcriptions : list
        A list of tuples containing the speaker's label and their corresponding transcription.

    Usage
    -----
    >>> transcript = Transcript("/path/to/audio.wav")
    >>> transcriptions = transcript.transcribe_audio()
    >>> transcript.save("/path/to/transcripts")
    """
    def __init__(self, audio_file: str):
        self.audio_file = audio_file
        self.name =  self.find_name()
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        if self.HF_TOKEN is None:
            raise ValueError("""HF_TOKEN not found in environment variables.
                             Try to run : export HF_TOKEN="your_hf_token""")
        self.transcriptions = []

    def find_name(self) -> str:
        pattern = r'/(.+?)\.wav'
        match = re.search(pattern, self.audio_file)
        if match:
            return match.group(1)
        return "transcript"

    def transcribe_audio(self) -> list:
        """
        Transcribes the audio file and returns a list of transcriptions.

        Returns
        -------
        list
            A list of tuples containing the speaker's label and their corresponding transcription.

        Raises
        ------
        RuntimeError
            If the audio file cannot be processed or a segment cannot be loaded.
        ValueError
            If the HF_TOKEN is incorrect or not found.
        Usage
        -----
        To transcribe the audio file, call the `transcribe_audio` method on an instance of the Transcript class.

        ```python
        transcriptions = transcript.transcribe_audio()
        ```
        """

        model = whisper.load_model("base")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=self.HF_TOKEN)
        audio = Audio()

        try:
            print("Processing audio file...")
            diarization = pipeline(self.audio_file)
        except Exception as e:
            raise RuntimeError(f"Failed to process the audio file: {e}")

        segments = list(diarization.itertracks(yield_label=True))
        transcriptions = []

        with tqdm.tqdm(total=len(segments), desc="Writing transcript", unit="segment", ncols=100, colour="green") as pbar:
            for turn, _, speaker in segments:
                segment = Segment(turn.start, turn.end)

                try:
                    waveform, sample_rate = audio.crop(self.audio_file, segment)
                except Exception as e:
                    raise RuntimeError(f"Failed to load the audio segment: {e}")

                waveform = waveform.numpy().flatten()
                if sample_rate != 16000:
                    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

                result = model.transcribe(waveform, fp16=False)
                transcriptions.append((speaker, result['text']))
                pbar.update(1)
        self.transcriptions = transcriptions
        for speaker, text in transcriptions:
            print(f"{speaker}: {text}")
            sleep(0.5)
        return self


    def save(self, directory: str =  "transcripts") -> None:
        """
        Saves the transcriptions to text files in the specified directory.

        Parameters
        ----------
        directory : str, optional
            A string containing the path to the directory where the transcriptions will be saved (default is "transcripts").

        Raises
        ------
        ValueError
            If no transcriptions are available to save.

        Usage
        -----
        To save the transcriptions to text files, call the `save` method on an instance of the Transcript class.

        ```python
        transcript.save("/path/to/transcripts")
        ```
        """
        if not self.transcriptions:
            raise ValueError("No transcriptions available to save.")
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, f"{self.name}_transcript.txt"), 'w') as f:
            for (speaker, text) in self.transcriptions:
                if text:
                    f.write(f"{speaker}: {text}\n")
