import os
from dotenv import load_dotenv
import whisper
from pyannote.audio import Audio, Pipeline
from pyannote.core import Segment
import librosa
import tqdm
from transcription import Transcription

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
        model = whisper.load_model("base")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=self.HF_TOKEN)
        audio = Audio()
        return model, pipeline, audio

    def transcribe_audio(self, audio_file_path: str) -> Transcription:
        try:
            print("Processing audio file...")
            diarization = self.pipeline(audio_file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to process the audio file: {e}")

        segments = list(diarization.itertracks(yield_label=True))
        transcriptions = []

        with tqdm.tqdm(total=len(segments), desc="Writing transcript", unit="segment", ncols=100, colour="green") as pbar:
            for turn, _, speaker in segments:
                segment = Segment(turn.start, turn.end)

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
                pbar.update(1)
        return Transcription(audio_file_path, transcriptions)


if __name__ == "__main__":
    transcriptor = Transcriptor()
    transcriptions = transcriptor.transcribe_audio(
        audio_file_path="audio-test/harvard.wav")
    print(transcriptions)
    transcriptions.save('.')
