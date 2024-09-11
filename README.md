# Audio Transcription and Diarization Tool

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Audio Processing Example](#audio-processing-example)
  - [Transcribing an existing audio file or an audio recording](#transcribing-an-existing-audio-file-or-an-audio-recording)
- [Key Components](#key-components)
  - [Transcriptor](#transcriptor)
  - [AudioProcessor](#audioprocessor)
  - [AudioRecording](#audiorecording)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

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
- `pyaudio`
- `tabulate`
- `soundfile`
- `torch`

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

## Usage

### Basic Example

Here's a basic example of how to use the Transcriptor class to transcribe an audio file:

```python
from pyscript import Transcriptor

# Initialize the Transcriptor
transcriptor = Transcriptor()

# Transcribe an audio file
transcription = transcriptor.transcribe_audio("/path/to/audio")

# Interactively name
transcription.get_name_speakers()

# Save the transcription
transcription.save()
```

This example will:
1. Initialize the Transcriptor with default settings
2. Transcribe the specified audio file
3. Prompt you to name the speakers interactively
4. Save the transcription to a text file

### Audio Processing Example

You can also use the AudioProcessor class to preprocess your audio files:

```python
from pyscript import AudioProcessor

# Load an audio file
audio = AudioProcessor("/path/to/audio.mp3")

# Display audio details
audio.display_details()

# Convert the audio to WAV format and resample to 16000 Hz
audio.convert_to_wav()

# Display updated audio details
audio.display_changes()
```

This example will:
1. Load an audio file
2. Display audio details
3. Convert the audio to WAV format and resample to 16000 Hz
4. Display updated audio details


### Transcribing an existing audio file or an audio recording

To transcribe an audio file or record and transcribe audio, run:

```bash
python example.py
```

You will be prompted to:
1. Choose between using an existing audio file or recording new audio
2. If using an existing file:
   - Select an audio file from the `audio-test` folder
   - Choose whether to transcribe one or all files
3. If recording new audio:
   - Provide a name for the audio file (optional)
   - Start recording

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
- Displaying audio file details and changes

### AudioRecording

The `audio_recording.py` module provides functions for recording audio from a microphone, checking input devices, and saving audio files.

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