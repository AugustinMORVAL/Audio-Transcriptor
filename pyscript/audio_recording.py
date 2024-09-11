import speech_recognition as sr
import os
import datetime
from termcolor import colored
from tabulate import tabulate

def micro_recording(save_folder_path: str = "audio_files", file_name: str = None, device_index: int = 0) -> str:
    """Records audio from a microphone and saves it to a designated file."""
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=device_index)

    print_colored_separator("Starting microphone recording...", "green")
    
    with mic as source:
        print_colored("Recording...", "yellow")
        audio = r.listen(source)
        print_colored("Recording finished.", "green")
    
    saved_path = save_audio_file(audio, save_folder_path, file_name)
    
    print_colored_separator(f"Audio file saved to: {saved_path}", "green")
    return saved_path

def check_input_device(test_duration: int = 1) -> dict:
    """Checks the available microphone devices."""
    devices = sr.Microphone.list_microphone_names()
    available_devices, non_working_devices = [], []

    for i, device in enumerate(devices):
        try:
            with sr.Microphone(device_index=i) as source:
                sr.Recognizer().listen(source, timeout=test_duration)
            available_devices.append(device)
        except sr.WaitTimeoutError:
            non_working_devices.append(device)
        except Exception as e:
            print(f"An error occurred while testing device {device}: {e}")

    print_device_table("Available Devices", available_devices)
    print_device_table("Non-Working Devices", non_working_devices)

    return {'available_devices': available_devices, 'non_working_devices': non_working_devices}

def save_audio_file(audio, save_folder_path: str, file_name: str = None) -> str:
    """Saves the audio file to the specified path."""
    os.makedirs(save_folder_path, exist_ok=True)
    
    if not file_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f"recording_{timestamp}.wav"
    else:
        file_name = f"{file_name}.wav"
    
    saved_path = os.path.join(save_folder_path, file_name)
    
    with open(saved_path, "wb") as f:
        f.write(audio.get_wav_data())
    
    print_colored("Saving audio file...", "yellow")
    return saved_path

def print_colored(message: str, color: str):
    """Prints a colored message."""
    print(colored(message, color))

def print_colored_separator(message: str, color: str):
    """Prints a colored message with separators."""
    print("--------------------------------")
    print_colored(message, color)
    print("--------------------------------")

def print_device_table(title: str, devices: list):
    """Prints a table of devices."""
    device_table = [[i+1, device] for i, device in enumerate(devices)]
    print(f"\n{title}:")
    print(tabulate(device_table, headers=["Index", "Device Name"]))

