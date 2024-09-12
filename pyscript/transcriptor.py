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
        The size of the Whisper model to use for transcription. Available options are:
        - 'tiny': Fastest, lowest accuracy
        - 'base': Fast, good accuracy for many use cases
        - 'small': Balanced speed and accuracy
        - 'medium': High accuracy, slower than smaller models
        - 'large': High accuracy, slower and more resource-intensive
        - 'large-v1': Improved version of the large model
        - 'large-v2': Further improved version of the large model
        - 'large-v3': Latest and most accurate version of the large model
    model : whisper.model.Whisper
        The Whisper model for transcription.
    pipeline : pyannote.audio.pipelines.SpeakerDiarization
        The PyAnnote speaker diarization pipeline.

    Usage:
        >>> transcript = Transcriptor(model_size="large-v3")
        >>> transcription = transcript.transcribe_audio("/path/to/audio.wav")
        >>> transcription.get_name_speakers()
        >>> transcription.save("/path/to/transcripts")

    Note:
        Larger models, especially 'large-v3', provide higher accuracy but require more 
        computational resources and may be slower to process audio.
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

    def transcribe_audio(self, audio_file_path: str, enhanced: bool = False) -> Transcription:
        """
        Transcribe an audio file.

        Parameters:
        -----------
        audio_file_path : str
            Path to the audio file to be transcribed.
        enhanced : bool, optional
            If True, applies audio enhancement techniques to improve transcription quality.
            This includes noise reduction, voice enhancement, and volume boosting.

        Returns:
        --------
        Transcription
            A Transcription object containing the transcribed text and speaker segments.
        """
        try:
            print("Processing audio file...")
            processed_audio = self.process_audio(audio_file_path, enhanced)
            audio_file_path = processed_audio.path
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

    def process_audio(self, audio_file_path: str, enhanced: bool = False) -> AudioProcessor:
        """
        Process the audio file to ensure it meets the requirements for transcription.

        Parameters:
        -----------
        audio_file_path : str
            Path to the audio file to be processed.
        enhanced : bool, optional
            If True, applies audio enhancement techniques to improve audio quality.
            This includes optimizing noise reduction, voice enhancement, and volume boosting
            parameters based on the audio characteristics.

        Returns:
        --------
        AudioProcessor
            An AudioProcessor object containing the processed audio file.
        """
        processed_audio = AudioProcessor(audio_file_path)
        if processed_audio.format != ".wav":
            processed_audio.convert_to_wav()
        if processed_audio.sample_rate != 16000:
            processed_audio.resample_wav()
        if enhanced:
            parameters = processed_audio.optimize_enhancement_parameters()
            processed_audio.enhance_audio(noise_reduce_strength=parameters[0], 
                                          voice_enhance_strength=parameters[1], 
                                          volume_boost=parameters[2])
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