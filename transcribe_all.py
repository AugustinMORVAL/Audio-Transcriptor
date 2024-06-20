from transcriptor import Transcriptor
import os

transcriptor = Transcriptor()
audio_files = os.listdir("audio-test")
for audio_file_path in audio_files:
    if audio_file_path.endswith(".wav"):
        try:
            transcriptions = transcriptor.transcribe_audio(
                audio_file_path=f"audio-test/{audio_file_path}")
            transcriptions.save()
        except Exception as e:
            print(f"Failed to transcribe {audio_file_path}: {e}")
