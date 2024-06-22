from pyscript import Transcriptor
from pyscript import audio_recording as recorder
import speech_recognition as sr

def ask_user() -> int:
    available_inputs = sr.Microphone.list_microphone_names()
    mics = {str(i): name for i, name in enumerate(available_inputs)}
    print("Select input device to use for recording :")
    for i, audio in mics.items():
        print(f"[{i}] {audio}")
    mic_index = int(input('Enter the number of the device : '))
    device = available_inputs[mic_index]
    print(f'You selected {device} as audio input.')
    return mic_index

transcriptor = Transcriptor()

filename = input("\nEnter the name of the audio file (Press Enter for default): ")
Input = input("Press Enter to start recording...")
if Input == "":
    recording = recorder.micro_recording(file_name=filename)

    transcription = transcriptor.transcribe_audio(recording)
    transcription.save()