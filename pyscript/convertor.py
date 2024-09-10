import os
from pydub import AudioSegment

class AudioConverter:
    """
    A class for converting various audio formats to WAV.

    This class uses pydub to convert different audio formats to WAV format.

    Usage
    -----
    >>> converter = AudioConverter()
    >>> wav_path = converter.convert_to_wav("/path/to/audio.m4a")
    """



    @staticmethod
    def _get_file_format(file_path: str) -> str:
        """
        Extracts the file format from the file path.

        Parameters
        ----------
        file_path : str
            The path to the file.

        Returns
        -------
        str
            The lowercase file extension without the dot.
        """
        return os.path.splitext(file_path)[1][1:].lower()