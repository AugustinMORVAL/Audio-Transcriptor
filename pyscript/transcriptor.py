import os
from dotenv import load_dotenv
import whisper
from pyannote.audio import Pipeline
import torch
from tqdm import tqdm
from time import time
from transformers import pipeline
from .transcription import Transcription
from .audio_processing import AudioProcessor
import io
from contextlib import redirect_stdout
import sys
from .utils import TqdmToLogger

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
        - 'large-v3-turbo': Optimized version of the large-v3 model for faster processing
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print("Initializing Whisper model...")
        if self.model_size == "large-v3-turbo":
            self.model = pipeline(
                task="automatic-speech-recognition",
                model="ylacombe/whisper-large-v3-turbo",
                chunk_length_s=30,
                device=self.device,
            )
        else:
            self.model = whisper.load_model(self.model_size, device=self.device)
        print("Building diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=self.HF_TOKEN
        ).to(torch.device(self.device))
        print("Setup completed successfully!")

    def transcribe_audio(self, audio_file_path: str, enhanced: bool = False, buffer_logs: bool = False):
        """
        Transcribe an audio file.

        Parameters:
        -----------
        audio_file_path : str
            Path to the audio file to be transcribed.
        enhanced : bool, optional
            If True, applies audio enhancement techniques to improve transcription quality.
        buffer_logs : bool, optional
            If True, captures logs and returns them with the transcription. If False, prints to terminal.

        Returns:
        --------
        Union[Transcription, Tuple[Transcription, str]]
            Returns either just the Transcription object (if buffer_logs=False) 
            or a tuple of (Transcription, logs string) if buffer_logs=True
        """
        if buffer_logs:
            # Create a string buffer to capture printed output
            logs_buffer = io.StringIO()
            # Redirect stdout to our buffer
            with redirect_stdout(logs_buffer):
                transcription = self._perform_transcription(audio_file_path, enhanced)
                logs = logs_buffer.getvalue()
                return transcription, logs
        else:
            # Direct terminal output
            transcription = self._perform_transcription(audio_file_path, enhanced)
            return transcription

    def _perform_transcription(self, audio_file_path: str, enhanced: bool = False):
        """Internal method to handle the actual transcription process."""
        try:
            print(f"Received audio_file_path: {audio_file_path}")
            print(f"Type of audio_file_path: {type(audio_file_path)}")
            
            if audio_file_path is None:
                raise ValueError("No audio file was uploaded. Please upload an audio file.")
            
            if not isinstance(audio_file_path, (str, bytes, os.PathLike)):
                raise ValueError(f"Invalid audio file path type: {type(audio_file_path)}")
            
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found at path: {audio_file_path}")
            
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
            print(f"Error occurred: {str(e)}")
            raise RuntimeError(f"Failed to process the audio file: {str(e)}")

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
        with torch.no_grad():
            return self.pipeline(audio_file_path)

    def transcribe_segments(self, audio, sr, duration, segments):
        """Transcribe audio segments based on diarization."""
        transcriptions = []
        
        # Get the current stdout buffer from the redirect_stdout context
        buf = tqdm._instances[0].fp if tqdm._instances else sys.stdout
        
        # Prepare all segments at once
        audio_segments = []
        for turn, _, speaker in segments:
            start = turn.start
            end = min(turn.end, duration)
            segment = audio[int(start * sr):int(end * sr)]
            audio_segments.append((segment, speaker))
        
        # Process segments with progress bar
        with tqdm(
            total=len(audio_segments),
            desc="Transcribing segments",
            unit="segment",
            ncols=100,
            colour="green",
            file=TqdmToLogger(buf),
            mininterval=0.1,
            dynamic_ncols=True,
            leave=True
        ) as pbar:
            if self.model_size == "large-v3-turbo" and self.device == "cuda":
                # Automatically determine batch size based on available GPU memory
                total_memory = torch.cuda.get_device_properties(0).total_memory
                reserved_memory = torch.cuda.memory_reserved(0)
                allocated_memory = torch.cuda.memory_allocated(0)
                free_memory = total_memory - reserved_memory - allocated_memory
                
                # Estimate memory per sample (conservative estimate: 500MB per audio segment)
                memory_per_sample = 500 * 1024 * 1024
                
                # Calculate batch size leaving 20% memory buffer
                batch_size = max(1, int((free_memory * 0.8) // memory_per_sample))
                print(f"Automatically set batch size to {batch_size} based on available GPU memory")
                
                # Process in batches
                for i in range(0, len(audio_segments), batch_size):
                    batch = audio_segments[i:i + batch_size]
                    results = self.model([segment for segment, _ in batch])
                    for (_, speaker), result in zip(batch, results):
                        transcriptions.append((speaker, result['text'].strip()))
                    pbar.update(len(batch))
            else:
                # Sequential processing for CPU or regular whisper model
                for segment, speaker in audio_segments:
                    if self.model_size == "large-v3-turbo":
                        result = self.model(segment)
                        transcriptions.append((speaker, result['text'].strip()))
                    else:
                        result = self.model.transcribe(segment, fp16=self.device == "cuda")
                        transcriptions.append((speaker, result['text'].strip()))
                    pbar.update(1)
        
        return transcriptions