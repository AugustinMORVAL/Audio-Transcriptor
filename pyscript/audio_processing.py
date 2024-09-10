import os
import librosa
from tabulate import tabulate
from pydub import AudioSegment

class AudioProcessor:

    def __init__(self, audio_file):
         self.path = audio_file
         self.name = os.path.splitext(os.path.basename(audio_file))[0]
         self.format = os.path.splitext(os.path.basename(audio_file))[1]
         self.duration = librosa.get_duration(filename=audio_file)
         self.sample_rate = librosa.get_samplerate(audio_file)

    def show_details(self):
        """
        Diplay the attributes of the audio file.

        Attributes:
        -----------
        name : str
        format : str
        duration : float
        sample_rate : int
        """
        data = [
            ["File Name", self.name],
            ["File Format", self.format],
            ["Duration", f"{self.duration} seconds"],
            ["Sample Rate", f"{self.sample_rate} Hz"]
        ]
        print(tabulate(data, headers=["Attribute", "Value"], tablefmt="outline"))

    def formating_for_transcription(self):
        pass

    def convert_to_wav(self):
        pass

        SUPPORTED_FORMATS = ['mp3', 'm4a', 'ogg', 'flac', 'aac', 'wma']
