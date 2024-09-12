import os
import librosa
import numpy as np
from tabulate import tabulate
import soundfile as sf
import scipy.ndimage
import itertools
from scipy.stats import pearsonr
from tqdm import tqdm

class AudioProcessor:

    def __init__(self, audio_file):
         self.path = audio_file
         self.name = os.path.splitext(os.path.basename(audio_file))[0]
         self.format = os.path.splitext(os.path.basename(audio_file))[1]
         self.duration = librosa.get_duration(path=audio_file)
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
        """Display the changes made to the audio file side by side."""
        self._clean_duplicates_changes()
        if len(self.changes) == 1:
            self.display_details()
        else:
            table1 = self.changes[0].split('\n')
            table2 = self.changes[-1].split('\n')

            combined_table = []
            for line1, line2 in zip(table1, table2):
                combined_table.append([line1, '===>', line2])

            print(tabulate(combined_table, tablefmt="plain"))

    def _clean_duplicates_changes(self):
        """Remove duplicate consecutive changes from the audio file."""
        self.changes = [change for i, change in enumerate(self.changes) 
                        if i == 0 or change != self.changes[i-1]]

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
        output_path = os.path.join('resampled_files', f'{self.name}.wav')
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

    def enhance_audio(self, noise_reduce_strength=0.75, voice_enhance_strength=2.0, volume_boost=1.5):
        """
        Enhance audio quality by reducing noise and clarifying voices.

        Parameters:
        -----------
        noise_reduce_strength : float, optional
            Strength of noise reduction (default is 0.5)
        voice_enhance_strength : float, optional
            Strength of voice enhancement (default is 1.2)
        volume_boost : float, optional
            Factor to boost the overall volume (default is 1.5)

        Returns:
        --------
        str
            Path to the enhanced audio file
        """
        try:
            # Load the audio file
            y, sr = librosa.load(self.path)

            # Noise reduction using spectral subtraction
            S = librosa.stft(y)
            S_mag = np.abs(S)
            S_phase = np.angle(S)
            S_filtered = scipy.ndimage.median_filter(S_mag, size=(1, 31))
            
            # Apply noise reduction
            mask = (S_mag - S_filtered) / (S_mag + 1e-10)
            mask = np.clip(mask, 0, 1)
            mask = mask ** noise_reduce_strength  # Apply noise reduction strength
            S_denoised = S_mag * mask * np.exp(1j * S_phase)

            # Inverse STFT
            y_denoised = librosa.istft(S_denoised)

            # Voice enhancement using harmonic-percussive source separation
            y_harmonic, y_percussive = librosa.effects.hpss(y_denoised)
            y_enhanced = y_harmonic * voice_enhance_strength + y_percussive

            # Apply volume boost
            y_enhanced = y_enhanced * volume_boost

            # Normalize audio to prevent clipping
            y_enhanced = librosa.util.normalize(y_enhanced, norm=np.inf, threshold=1.0)

            # Save the enhanced audio
            output_path = os.path.join('enhanced_files', f'{self.name}_enhanced.wav')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, y_enhanced, sr)

            self._update_file_info(output_path)
            return output_path
        except Exception as e:
            raise RuntimeError(f"Failed to enhance audio: {e}")
        
    def optimize_enhancement_parameters(self, step=0.25, max_iterations=50, sample_duration=30):
        """
        Find optimal parameters for audio enhancement using grid search on a 30 seconde sample.
        
        Parameters:
        -----------
        step : float, optional
            Step size for parameter values (default is 0.25)
        max_iterations : int, optional
            Maximum number of iterations for the grid search (default is 50)
        sample_duration : int, optional
            Duration of the audio sample in seconds (default is 60)

        Returns:
        --------
        float, float, float
            Optimal values for noise_reduce_strength, voice_enhance_strength, volume_boost
        """
        
        y_orig, sr = librosa.load(self.path, duration=sample_duration)
        
        # Define parameter ranges
        param_ranges = [
            np.arange(0.25, 1.5, step),  # noise_reduce_strength
            np.arange(1.0, 3.0, step),   # voice_enhance_strength
            np.arange(1.0, 2.0, step)    # volume_boost
        ]

        best_score = float('-inf')
        best_params = None
        
        # Calculate total iterations
        total_iterations = min(max_iterations, len(list(itertools.product(*param_ranges))))
        
        # Perform grid search with tqdm progress bar
        for params in tqdm(itertools.islice(itertools.product(*param_ranges), max_iterations), 
                           total=total_iterations, 
                           desc="Searching for optimal parameters"):
            # Apply enhancement with current parameters
            y_enhanced = self._enhance_audio_sample(y_orig, sr, *params)
            
            # Ensure both arrays have the same length
            min_length = min(len(y_orig), len(y_enhanced))
            y_orig_trimmed = y_orig[:min_length]
            y_enhanced_trimmed = y_enhanced[:min_length]
            
            # Calculate correlation between original and enhanced audio
            correlation, _ = pearsonr(y_orig_trimmed, y_enhanced_trimmed)
            
            # Calculate spectral contrast
            S_orig = np.abs(librosa.stft(y_orig_trimmed))
            S_enhanced = np.abs(librosa.stft(y_enhanced_trimmed))
            contrast_orig = np.mean(librosa.feature.spectral_contrast(S=S_orig))
            contrast_enhanced = np.mean(librosa.feature.spectral_contrast(S=S_enhanced))
            contrast_improvement = contrast_enhanced - contrast_orig
            
            # Calculate combined score (you can adjust the weights)
            score = correlation + 0.5 * contrast_improvement
            
            if score > best_score:
                best_score = score
                best_params = params

        print(f"\nBest parameters found: {best_params}")
        noise_reduce_strength, voice_enhance_strength, volume_boost = best_params
        return noise_reduce_strength, voice_enhance_strength, volume_boost

    def _enhance_audio_sample(self, y, sr, noise_reduce_strength, voice_enhance_strength, volume_boost):
        # Noise reduction using spectral subtraction
        S = librosa.stft(y)
        S_mag = np.abs(S)
        S_phase = np.angle(S)
        S_filtered = scipy.ndimage.median_filter(S_mag, size=(1, 31))
        
        # Apply noise reduction
        mask = (S_mag - S_filtered) / (S_mag + 1e-10)
        mask = np.clip(mask, 0, 1)
        mask = mask ** noise_reduce_strength  # Apply noise reduction strength
        S_denoised = S_mag * mask * np.exp(1j * S_phase)

        # Inverse STFT
        y_denoised = librosa.istft(S_denoised)

        # Voice enhancement using harmonic-percussive source separation
        y_harmonic, y_percussive = librosa.effects.hpss(y_denoised)
        y_enhanced = y_harmonic * voice_enhance_strength + y_percussive

        # Apply volume boost
        y_enhanced = y_enhanced * volume_boost

        # Normalize audio to prevent clipping
        y_enhanced = librosa.util.normalize(y_enhanced, norm=np.inf, threshold=1.0)

        return y_enhanced

    # Helper method
    def _update_file_info(self, new_path):
        """Update file information after processing."""
        self.path = new_path
        self.sample_rate = librosa.get_samplerate(new_path)
        self.format = os.path.splitext(new_path)[1]
        self.load_details()