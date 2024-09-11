from pyscript import Transcriptor
from pyscript import audio_recording as recorder
import os

def choose_audio_usage(): 
    print("Choose audio usage:")
    print("[1] From file")
    print("[2] From recording")
    choice = input("Enter the number of the audio type: ")
    if choice == "1":
        return True
    elif choice == "2":
        return False
    else: 
        return choose_audio_usage()

def process_audio(file_path, transcriptor):
    try:
        transcriptions = transcriptor.transcribe_audio(file_path)
        transcriptions.get_name_speakers()
        transcriptions.save()
        return transcriptions
    except Exception as e:
        print(f"Failed to process {file_path}: {str(e)}")

def main():
    transcriptor = Transcriptor()

    audio_usage = choose_audio_usage()
    if audio_usage:
        folder = "audio-test"

        print("\nSelect audio file to transcribe:")
        audio_files = [f for f in os.listdir(folder)]
        for i, audio in enumerate(audio_files, 1):
            print(f"[{i}] {audio}")
        print(f"[{len(audio_files) + 1}] All")

        choice = input('Enter the number of the audio file to transcribe: ')

        if choice == str(len(audio_files) + 1):
            for file in audio_files:
                process_audio(os.path.join(folder, file), transcriptor)
        if int(choice) > 0 and int(choice) <= len(audio_files):
            file_path = os.path.join(folder, audio_files[int(choice) - 1])
            transcriptions = process_audio(file_path, transcriptor)
            if transcriptions:
                print(f'Transcrriptions:\n {transcriptions}')
    else:
        filename = input("Enter the name of the audio file (Press Enter for default): ")
        Input = input("Press Enter to start recording...")
        if Input == "":
            recording = recorder.micro_recording(file_name=filename)

            transcriptions = process_audio(recording, transcriptor)
            if transcriptions:
                print(transcriptions)
                
if __name__ == "__main__":
    main()
