import os
from dotenv import load_dotenv
import whisper
from pyannote.audio import Pipeline
import librosa
from tqdm import tqdm
from .transcription import Transcription
from time import time
import spacy

# Load environment variables from .env file
load_dotenv()

class Transcriptor:
    """
    A class for transcribing and diarizing audio files.

    This class uses the Whisper model for transcription and the PyAnnote speaker diarization pipeline for speaker identification.

    Attributes
    ----------
    HF_TOKEN : str
        The Hugging Face token for accessing the PyAnnote speaker diarization pipeline.
    model : whisper.model.Whisper
        The Whisper model for transcription.
    pipeline : pyannote.audio.pipelines.SpeakerDiarization
        The PyAnnote speaker diarization pipeline.

    Usage
    -----
    >>> transcript = Transcriptor()
    >>> transcription = transcript.transcribe_audio("/path/to/audio.wav")
    >>> transcription.save("/path/to/transcripts")
    """

    def __init__(self):
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        if self.HF_TOKEN is None:
            raise ValueError("HF_TOKEN not found. Please store token in .env")
        self.model, self.pipeline = self.initialize_models()

    def initialize_models(self):
        """
        Initializes the Whisper model and the PyAnnote speaker diarization pipeline.

        Returns
        -------
        model : whisper.model.Whisper
            The Whisper model for transcription.
        pipeline : pyannote.audio.pipelines.SpeakerDiarization
            The PyAnnote speaker diarization pipeline.
        """
        print("Initializing Whisper model...")
        model = whisper.load_model("base")
        print("Building diarization pipeline...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=self.HF_TOKEN)
        print("Models initialized successfully.")
        return model, pipeline

    def transcribe_audio(self, audio_file_path: str) -> Transcription:
        """
        Transcribes an audio file.

        Parameters
        ----------
        audio_file_path : str
            The path to the audio file to be transcribed.

        Returns
        -------
        transcription : Transcription
            A Transcription object containing the speaker's label and their corresponding transcription.
        """
        # Diarization
        try:
            print("Processing audio file...")
            top = time()
            diarization = self.pipeline(audio_file_path)
            print(f"Audio file processed successfully in {time()-top}.")
        except Exception as e:
            raise RuntimeError(f"Failed to process the audio file: {e}")
        segments = list(diarization.itertracks(yield_label=True))

        # Transcription
        transcriptions = []
        duration = librosa.get_duration(filename=audio_file_path)
        for turn, _, speaker in tqdm(segments, desc="Writing transcript", unit="segment", ncols=100, colour="green"):
            start = turn.start
            if turn.end >= duration:
                end = duration
            else:
                end = turn.end
            try:
                waveform, sample_rate = librosa.load(audio_file_path, sr=16000, offset=start, duration=end - start)
            except Exception as e:
                raise RuntimeError(f"Failed to load the audio segment: {e}")
            
            result = self.model.transcribe(waveform, fp16=False)
            transcriptions.append((speaker, result['text']))
        return Transcription(audio_file_path, transcriptions)
