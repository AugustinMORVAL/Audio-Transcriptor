from pyscript import Transcriptor
import os

def process_audio(file_path, transcriptor, save=False):
    try:
        print(f"Processing file: {file_path}")
        transcriptions = transcriptor.transcribe_audio(file_path)
        transcriptions.name_speakers_interactively()
        if save:
            transcriptions.save()
        return transcriptions
    except Exception as e:
        print(f"Failed to process {file_path}: {str(e)}")
        return None

def main():
    transcriptor = Transcriptor()

    folder = "audio-test"

    print("Select audio file to transcribe:")
    audio_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    for i, audio in enumerate(audio_files, 1):
        print(f"[{i}] {audio}")
    print(f"[{len(audio_files) + 1}] All")

    choice = input('Enter the number of the audio file to transcribe: ')
    save = input("Do you want to save transcripts? (y/n): ").strip().lower() == 'y'

    if choice == str(len(audio_files) + 1):
        for file in audio_files:
            process_audio(os.path.join(folder, file), transcriptor, save)
    else:
        file_path = os.path.join(folder, audio_files[int(choice) - 1])
        transcriptions = process_audio(file_path, transcriptor, save)
        if transcriptions:
            print(transcriptions)

if __name__ == "__main__":
    main()
