# Audio Transcription and Diarization Tool

## Overview

This project provides a robust set of tools for transcribing audio files using the Whisper model and performing speaker diarization with PyAnnote. Users can process audio files, record audio, and save transcriptions with speaker identification.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Audio Processing Example](#audio-processing-example)
  - [Transcribing an Existing Audio File or Recording](#transcribing-an-existing-audio-file-or-recording)
- [Key Components](#key-components)
  - [Transcriptor](#transcriptor)
  - [AudioProcessor](#audioprocessor)
  - [AudioRecording](#audiorecording)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Features

- **Transcription**: Convert audio files in various formats to text (automatically converts to WAV).
- **Speaker Diarization**: Identify different speakers in the audio.
- **Speaker Retrieval**: Name speakers during transcription.
- **Audio Recording**: Record audio directly from a microphone.
- **Audio Preprocessing**: Includes resampling, format conversion, and audio enhancement.
- **Multiple Model Support**: Choose from various Whisper model sizes.

## Supported Whisper Models

This tool supports various Whisper model sizes, allowing you to balance accuracy and computational resources:

- **`tiny`**: Fastest, lowest accuracy
- **`base`**: Fast, good accuracy
- **`small`**: Balanced speed and accuracy
- **`medium`**: High accuracy, slower
- **`large`**: High accuracy, resource-intensive
- **`large-v1`**: Improved large model
- **`large-v2`**: Further improved large model
- **`large-v3`**: Latest and most accurate
- **`large-v3-turbo`**: Optimized for faster processing

Specify the model size when initializing the Transcriptor:

```python
transcriptor = Transcriptor(model_size="base")
```

The default model size is "base" if not specified.

## Requirements

To run this project, you need Python 3.7+ and the following packages:

```plaintext
- openai-whisper
- pyannote.audio
- librosa
- tqdm
- python-dotenv
- termcolor
- pydub
- SpeechRecognition
- pyaudio
- tabulate
- soundfile
- torch
- numpy
- transformers
- gradio
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/audio-transcription-tool.git
   cd audio-transcription-tool
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment variables**:
   - Create a `.env` file in the root directory.
   - Add your Hugging Face token:
     ```plaintext
     HF_TOKEN=your_hugging_face_token_here
     ```

## Usage

### Basic Example

Here's how to use the Transcriptor class to transcribe an audio file:

```python
from pyscript import Transcriptor

# Initialize the Transcriptor
transcriptor = Transcriptor()

# Transcribe an audio file
transcription = transcriptor.transcribe_audio("/path/to/audio")

# Interactively name speakers
transcription.get_name_speakers()

# Save the transcription
transcription.save()
```

### Audio Processing Example

Use the AudioProcessor class to preprocess your audio files:

```python
from pyscript import AudioProcessor

# Load an audio file
audio = AudioProcessor("/path/to/audio.mp3")

# Display audio details
audio.display_details()

# Convert to WAV format and resample to 16000 Hz
audio.convert_to_wav()

# Display updated audio details
audio.display_changes()
```

### Transcribing an Existing Audio File or Recording

To transcribe an audio file or record and transcribe audio, use the demo application provided in `demo.py`:

```bash
python demo.py
```

## Key Components

### Transcriptor

The `Transcriptor` class (in `pyscript/transcriptor.py`) is the core of the transcription process. It handles:

- Loading the Whisper model
- Setting up the diarization pipeline
- Processing audio files
- Performing transcription and diarization

### AudioProcessor

The `AudioProcessor` class (in `pyscript/audio_processing.py`) manages audio file preprocessing, including:

- Loading audio files
- Resampling
- Converting to WAV format
- Displaying audio file details and changes
- Audio enhancement (noise reduction, voice enhancement, volume boost)

### AudioRecording

The `audio_recording.py` module provides functions for recording audio from a microphone, checking input devices, and saving audio files.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request

## Acknowledgments

- OpenAI for the Whisper model
- PyAnnote for the speaker diarization pipeline
- All contributors and users of this project