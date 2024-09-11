# Audio Transcription and Diarization Tool

## Overview

This project provides a robust set of tools for transcribing audio files using the Whisper model and performing speaker diarization with PyAnnote. It allows users to process audio files, record audio, and save transcriptions with speaker identification.

## Features

- Transcribe audio files in various formats (automatically converts to WAV)
- Perform speaker diarization to identify different speakers in the audio
- Interactive speaker naming during transcription
- Record audio directly from a microphone
- Save transcriptions to text files
- Support for different Whisper model sizes
- Audio preprocessing capabilities (resampling, format conversion)

## Requirements

To run this project, you need Python 3.7+ and the following packages:

- `openai-whisper`
- `pyannote.audio`
- `librosa`
- `tqdm`
- `python-dotenv`
- `termcolor`
- `pydub`
- `SpeechRecognition`

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/audio-transcription-tool.git
   cd audio-transcription-tool
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file in the root directory
   - Add your Hugging Face token:
     ```bash
     HF_TOKEN=your_hugging_face_token_here
     ```

## Example of usage

### Transcribing Audio Files
To transcribe an audio file, run the following command:

```bash
python examples/transcribe.py
```

You will be prompted to:
1. Select an audio file from the `audio-test` folder
2. Choose whether to save the transcript

### Recording and Transcribing Audio
To record audio and transcribe it, use:

```bash
python examples/record.py
```

You will be prompted to:
1. Select an input device
2. Provide a name for the audio file (optional)
3. Start recording

## Key Components

### Transcriptor

The `Transcriptor` class (in `pyscript/transcriptor.py`) is the core of the transcription process. It handles:
- Loading the Whisper model
- Setting up the diarization pipeline
- Processing audio files
- Performing transcription and diarization

Usage:

```python
from pyscript import Transcriptor
transcriptor = Transcriptor(model_size="base")
transcription = transcriptor.transcribe_audio("/path/to/audio.wav")
transcription.name_speakers_interactively()
transcription.save()
```

### AudioProcessor

The `AudioProcessor` class (in `pyscript/audio_processing.py`) handles audio file preprocessing, including:
- Loading audio files
- Resampling
- Converting to WAV format

### AudioRecording

The `audio_recording.py` module provides functions for recording audio from a microphone and checking input devices.

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request

## Acknowledgments

- OpenAI for the Whisper model
- PyAnnote for the speaker diarization pipeline
- All contributors and users of this project