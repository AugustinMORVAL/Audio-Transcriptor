import os
import librosa
from tabulate import tabulate

class Processor:

    def __init__(self, audio_file):
         self.path = audio_file
         self.name = os.path.splitext(os.path.basename(audio_file))[0]
         self.format = os.path.splitext(os.path.basename(audio_file))[1]
         self.duration = librosa.get_duration(filename=audio_file)
         self.sample_rate = librosa.get_samplerate(audio_file)  

    def show_details(self):
        data = [
            ["File Name", self.name],
            ["File Format", self.format],
            ["Duration", f"{self.duration} seconds"],
            ["Sample Rate", f"{self.sample_rate} Hz"]
        ]
        print(tabulate(data, headers=["Attribute", "Value"], tablefmt="outline"))