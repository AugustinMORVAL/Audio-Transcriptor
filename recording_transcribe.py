from pyscript import Transcriptor
from pyscript import Audio
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
    print(mic_index)
    return mic_index

transcriptor = Transcriptor()
recorder = Audio()


device_index =  ask_user()

Input = input("Press Enter to start recording...")
if Input == "":
    recording = recorder.micro_recording(device_index=device_index)

    transcription = transcriptor.transcribe_audio(recording)
    transcription.save()