import os
from dotenv import load_dotenv
import whisper
from pyannote.audio import Pipeline
import librosa
from tqdm import tqdm
from .transcription import Transcription
from .audio_processing import AudioProcessor
from time import time
import torch

# Load environment variables from .env file
load_dotenv()

class Transcriptor:
    """
    A class for transcribing and diarizing audio files.

    This class uses the Whisper model for transcription and the PyAnnote speaker diarization pipeline for speaker identification.

    Attributes
    ----------
    model_size : str
        The size of the Whisper model to use for transcription.
    model : whisper.model.Whisper
        The Whisper model for transcription.
    pipeline : pyannote.audio.pipelines.SpeakerDiarization
        The PyAnnote speaker diarization pipeline.

    Usage
    -----
    >>> transcript = Transcriptor(model_size="large")  # Specify model size here
    >>> transcription = transcript.transcribe_audio("/path/to/audio.wav")
    >>> transcription.save("/path/to/transcripts")
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        if not self.HF_TOKEN:
            raise ValueError("HF_TOKEN not found. Please store token in .env")
        self.setup()

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model_name = self.model_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Initializing Whisper model...")
        self.model = whisper.load_model(
            model_name,
            device=device)
        print("Building diarization pipeline...")
        self.diarization_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=self.HF_TOKEN).to(torch.device(device))
        print("Setup completed successfully!")

    def transcribe_audio(self, audio_file_path: str) -> Transcription:
        """
        Transcribes an audio file.
        """
        try:
            print("Processing audio file...")
            start_time = time()

            # audio_file_path = self.convert_to_wav(audio_file_path)
            audio, sr, duration = self.load_audio(audio_file_path)
            diarization = self.perform_diarization(audio_file_path)
            segments = list(diarization.itertracks(yield_label=True))
            print(f"Audio file processed successfully in {time() - start_time:.2f} seconds.")
        except Exception as e:
            raise RuntimeError(f"Failed to process the audio file: {e}")

        transcriptions = self.transcribe_segments(audio, sr, duration, diarization)
        return Transcription(audio_file_path, transcriptions, segments)

    def convert_to_wav(self, audio_file_path: str) -> str:
        """Convert audio to WAV format."""
        file_extension = os.path.splitext(audio_file_path)[1].lower()
        if file_extension != '.wav':
            wav_file_path = os.path.splitext(audio_file_path)[0] + '.wav'
            audio_file_path = AudioProcessor.convert_to_wav(audio_file_path, wav_file_path)
            print(f"Converted audio file to WAV: {audio_file_path}")
        return audio_file_path

    def load_audio(self, audio_file_path: str):
        """Load audio file and return audio data, sample rate, and duration."""
        audio, sr = librosa.load(audio_file_path, sr=16000)
        duration = librosa.get_duration(y=audio, sr=sr)
        return audio, sr, duration

    def perform_diarization(self, audio_file_path: str):
        """Perform speaker diarization on the audio file."""
        return self.pipeline(audio_file_path)

    def transcribe_segments(self, audio, sr, duration, segments):
        """Transcribe audio segments based on diarization."""
        transcriptions = []
        for turn, _, speaker in tqdm(segments, desc="Transcribing segments", unit="segment", ncols=100, colour="green"):
            start = turn.start
            end = min(turn.end, duration)
            segment = audio[int(start * sr):int(end * sr)]
            result = self.model.transcribe(segment, fp16=True)
            transcriptions.append((speaker, result['text'].strip()))
        return transcriptions