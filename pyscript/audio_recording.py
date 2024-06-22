import speech_recognition as sr
import os
import datetime
from tqdm import tqdm
from termcolor import colored
from tabulate import tabulate

def micro_recording(save_folder_path: str = "audio_files", file_name: str = None, device_index: int = 0) -> str:
    """
    Records audio from a microphone and saves it to a designated file (optional).

    Parameters
    ----------
    save_folder_path : str, optional
        The path to the folder where the audio file will be saved. Defaults to "audio_files".
    file_name : str, optional
        The name of the audio file. If not provided, a timestamp will be used.
    device_index : int, optional
        The index of the microphone device to use. Defaults to 0.

    Returns
    -------
    str
        The path of the saved audio file.

    Usage
    -----
    >>> saved_path = micro_recording()
    >>> print(saved_path)
    """
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=device_index)
    print("--------------------------------")
    print(colored("Starting microphone recording...", "green"))
    with mic as source:
        print(colored("Recording...", "yellow"))
        audio = r.listen(source)
        print(colored("Recording finished.", "green"))
    print("--------------------------------")
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    if not file_name:
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S")
        file_name = f"recording_{timestamp_str}.wav"
    else:
        file_name = f"{file_name}.wav"
    saved_path = os.path.join(save_folder_path, file_name)
    with open(saved_path, "wb") as f:
        f.write(audio.get_wav_data())
    print("--------------------------------")
    print(colored("Saving audio file...", "yellow"))
    print(colored(f"Audio file saved to: {saved_path}", "green"))
    print("--------------------------------\n")
    return saved_path

def check_input_device(test_duration=1) -> dict:
    """
    Checks the available microphone devices.

    Parameters
    ----------
    test_duration : int, optional
        The duration of the listening test, in seconds. Default is 1.

    Returns
    -------
    dict
        A dictionary containing two lists: 'available_devices' and 'non_working_devices'.

    Usage
    -----
    >>> check_input_device()
    """
    devices = sr.Microphone.list_microphone_names()
    available_devices = []
    non_working_devices = []
    for i, item in enumerate(devices):
        try:
            r = sr.Microphone(device_index=i)
            with sr.Microphone() as source:
                r.listen(source, timeout=test_duration)
            if item not in available_devices:
                available_devices.append(item)
        except sr.WaitTimeoutError:
            if item not in non_working_devices:
                non_working_devices.append(item)
        except Exception as e:
            print(f"An error occurred while testing device {item}: {e}")

    # Table for available devices
    available_devices_table = [[i+1, device] for i, device in enumerate(available_devices)]
    print("\nAvailable Devices:")
    print(tabulate(available_devices_table, headers=["Index", "Device Name"]))

    # Table for non-working devices
    non_working_devices_table = [[i+1, device] for i, device in enumerate(non_working_devices)]
    print("\nNon-Working Devices:")
    print(tabulate(non_working_devices_table, headers=["Index", "Device Name"]))

    return {'available_devices': available_devices, 'non_working_devices': non_working_devices}

