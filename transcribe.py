from transcriptor import Transcriptor
import os


def ask_user(folder):
    audio_files = os.listdir(folder)
    audios = {str(i+1): name for i, name in enumerate(audio_files)
              if name.endswith('.wav')}
    print("Select the audio file to transcribe:")
    for i, audio in audios.items():
        print(f"[{i}] {audio}")
    file_index = input('Enter the number of the audio file to transcribe: ')
    audio_file = audios[file_index]
    audio_file_path = os.path.join(folder, audio_file)
    return audio_file_path


transcriptor = Transcriptor()
folder = "audio-test"
audio_file_path = ask_user(folder)

try:
    transcriptions = transcriptor.transcribe_audio(
        audio_file_path=audio_file_path)
    print(transcriptions)
    if input("Do you want to save (y/n)? ").strip().lower() == 'y':
        transcriptions.save()
except Exception as e:
    print(f"Failed to transcribe1 {audio_file_path}: {e}")
