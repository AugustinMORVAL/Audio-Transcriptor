import os
from dotenv import load_dotenv
import whisper
from pyannote.audio import Audio, Pipeline
from pyannote.core import Segment
import librosa
from tqdm import tqdm
from transcription import Transcription
from time import time
# Load environment variables from .env file
load_dotenv()


class Transcriptor:
    def __init__(self):
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        if self.HF_TOKEN is None:
            raise ValueError(
                "HF_TOKEN not found in environment variables. Try to run: export HF_TOKEN='your_hf_token'")
        self.model, self.pipeline, self.audio = self.initialize_whisper()

    def initialize_whisper(self):
        print("Initializing whisper...")
        model = whisper.load_model("base")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=self.HF_TOKEN)
        audio = Audio()
        print("Whisper initialized successfully .")
        return model, pipeline, audio

    def transcribe_audio(self, audio_file_path: str) -> Transcription:
        try:
            print("Processing audio file...")
            top = time()
            diarization = self.pipeline(audio_file_path)
            print(f"Audio file processed successfully in {time()-top}.")
        except Exception as e:
            raise RuntimeError(f"Failed to process the audio file: {e}")

        segments = list(diarization.itertracks(yield_label=True))
        transcriptions = []
        duration = librosa.get_duration(filename=audio_file_path)
        for turn, _, speaker in tqdm(segments, desc="Writing transcript", unit="segment", ncols=100, colour="green"):
            start = turn.start
            if turn.end >= duration:
                end = duration
            else:
                end = turn.end
            segment = Segment(start, end)
            try:
                waveform, sample_rate = self.audio.crop(
                    audio_file_path, segment)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load the audio segment: {e}")

            waveform = waveform.numpy().flatten()
            if sample_rate != 16000:
                waveform = librosa.resample(
                    waveform, orig_sr=sample_rate, target_sr=16000)

            result = self.model.transcribe(waveform, fp16=False)
            transcriptions.append((speaker, result['text']))
        return Transcription(audio_file_path, transcriptions)


if __name__ == "__main__":
    transcriptor = Transcriptor()
    transcriptions = transcriptor.transcribe_audio(
        audio_file_path="audio-test/meeting-clip1.wav")
    print(transcriptions)
