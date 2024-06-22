import speech_recognition as sr
import os
import datetime

def check_available_device() -> list:
    """Check the available microphone devices.
    Usage: available_devices, non_working_devices = r.check_available_device()"""
    devices = sr.Microphone.list_microphone_names()
    available_devices = []
    non_working_devices = []
    for i, item in enumerate(devices): 
        try:
            sr.Microphone(device_index=i)
            if item not in available_devices : 
                available_devices.append(item)
        except:
            if item not in non_working_devices :
                non_working_devices.append(item)
    if not non_working_devices :
        print("All devices are available.")
        print("Number of available devices: ", len(available_devices))
    else :
        print("Number of non-working devices: ", len(non_working_devices))
        print("Those devices are not available :") 
        for item in non_working_devices: 
                print(item)
    return available_devices, non_working_devices


def micro_recording(save_folder_path :str = "audio_files", file_name :str = None, device_index : int = 0) ->  str:
    """Record from microphone and save to a designated file (optional).
    Return  the path of the saved audio file."""
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=device_index)
    with mic as source:
        print("Recording...")
        audio = r.listen(source)
        print("Recording finished.")
        
        if not os.path.exists(save_folder_path):
            print(f"No {save_folder_path} folder found.")
            print(f"Creating folder: {save_folder_path}")
            os.makedirs(save_folder_path)

        if not file_name:
            timestamp = datetime.datetime.now("%Y%m%d-%H%M%S")
            file_name = f"recording_{timestamp}.wav"
        else : 
            file_name = f"{file_name}.wav"
        saved_path = os.path.join(save_folder_path, file_name)
        print("Saving file...")
        with open(saved_path, "wb") as f:
            f.write(audio.get_wav_data())
    print(f"Audio file saved to {saved_path}")
    return saved_path