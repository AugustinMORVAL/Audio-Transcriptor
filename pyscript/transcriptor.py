import os
from dotenv import load_dotenv
import whisper
from pyannote.audio import Pipeline
import torch
from tqdm import tqdm
from time import time
from .transcription import Transcription
from .audio_processing import AudioProcessor

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

    Usage:
        >>> transcript = Transcriptor(model_size="large")
        >>> transcription = transcript.transcribe_audio("/path/to/audio.wav")
        >>> transcription.get_name_speakers()
        >>> transcription.save("/path/to/transcripts")
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        if not self.HF_TOKEN:
            raise ValueError("HF_TOKEN not found. Please store token in .env")
        self._setup()

    def _setup(self):
        """Initialize the Whisper model and diarization pipeline."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Initializing Whisper model...")
        self.model = whisper.load_model(self.model_size, device=device)
        print("Building diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=self.HF_TOKEN
        ).to(torch.device(device))
        print("Setup completed successfully!")

    def transcribe_audio(self, audio_file_path: str) -> Transcription:
        """Transcribe an audio file."""
        try:
            print("Processing audio file...")
            processed_audio = self.process_audio(audio_file_path)
            audio, sr, duration = processed_audio.load_as_array(), processed_audio.sample_rate, processed_audio.duration
            
            print("Diarization in progress...")
            start_time = time()
            diarization = self.perform_diarization(audio_file_path)
            print(f"Diarization completed in {time() - start_time:.2f} seconds.")
            segments = list(diarization.itertracks(yield_label=True))
            
            transcriptions = self.transcribe_segments(audio, sr, duration, segments)
            return Transcription(audio_file_path, transcriptions, segments)
        except Exception as e:
            raise RuntimeError(f"Failed to process the audio file: {e}")

    def process_audio(self, audio_file_path: str) -> AudioProcessor:
        """Process the audio file to ensure it meets the requirements for transcription."""
        processed_audio = AudioProcessor(audio_file_path)
        if processed_audio.format != ".wav":
            processed_audio.convert_to_wav()
        if processed_audio.sample_rate != 16000:
            processed_audio.resample()
        processed_audio.display_changes()
        return processed_audio

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