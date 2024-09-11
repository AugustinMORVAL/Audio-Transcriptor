import os
import librosa
import numpy as np
from tabulate import tabulate
import soundfile as sf

class AudioProcessor:

    def __init__(self, audio_file):
         self.path = audio_file
         self.name = os.path.splitext(os.path.basename(audio_file))[0]
         self.format = os.path.splitext(os.path.basename(audio_file))[1]
         self.duration = librosa.get_duration(filename=audio_file)
         self.sample_rate = librosa.get_samplerate(audio_file)
         self.changes = []
         self.load_details()

    # File information methods
    def load_details(self):
        """Save the attributes of the audio file."""
        data = [
            ["File Name", self.name],
            ["File Format", self.format],
            ["Duration", f"{self.duration} seconds"],
            ["Sample Rate", f"{self.sample_rate} Hz"]
        ]
        table = tabulate(data, headers=["Attribute", "Value"], tablefmt="outline")
        self.changes.append(table)
        return table
    
    def display_details(self):
        """Display the details of the audio file."""
        print(self.changes[-1])

    def display_changes(self):
        """Display the changes made to the audio file."""
        data1 = self.changes[0]
        data2 = self.changes[-1]
        print(f"{data1} ===> {data2}")

    # Audio processing methods
    def load_as_array(self, sample_rate: int = 16000) -> np.ndarray:
        """
        Load an audio file and convert it into a NumPy array.

        Parameters
        ----------
        sample_rate : int, optional
            The sample rate to which the audio will be resampled (default is 16000 Hz).

        Returns
        -------
        np.ndarray
            A NumPy array containing the audio data.
        """
        try:
            audio, sr = librosa.load(self.path, sr=sample_rate)
            self.sample_rate = sr
            return audio
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")
        
    def resample_wav(self) -> str:
        output_path = os.path.join('resampled_files', f'{self.name}_resampled.wav')
        try:
            audio, sr = librosa.load(self.path)
            resampled_audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, resampled_audio, 16000)
            self._update_file_info(output_path)
            return output_path
        except Exception as e:
            raise RuntimeError(f"Failed to resample audio file: {e}")
        
    def convert_to_wav(self):
        """
        Converts an audio file to WAV format.

        Returns
        -------
        str
            The path to the converted audio file.
        """        
        output_path = os.path.join('converted_files', f'{self.name}.wav')
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            audio, sr = librosa.load(self.path)
            resampled_audio = librosa.resample(y=audio, orig_sr=sr, target_sr=16000)
            sf.write(output_path, resampled_audio, 16000)
            self._update_file_info(output_path)
            return output_path
        except Exception as e:
            raise RuntimeError(f"Failed to convert audio file to WAV: {e}")

    # Helper method
    def _update_file_info(self, new_path):
        """Update file information after processing."""
        self.path = new_path
        self.sample_rate = librosa.get_samplerate(new_path)
        self.format = os.path.splitext(new_path)[1]
        self.load_details()