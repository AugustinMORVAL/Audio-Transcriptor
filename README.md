# Audio Transcription Tool

## Overview
This project provides a set of tools for transcribing audio files using the Whisper model and performing speaker diarization with PyAnnote. It allows users to process audio files, record audio, and save transcriptions.

## Features
- Transcribe audio files in WAV format.
- Interactive speaker naming during transcription.
- Record audio directly from a microphone.
- Save transcriptions to text files.

## Requirements
To run this project, you need the following Python packages:

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

## Example of usage

### Transcribing Audio Files
To transcribe an audio file, run the following command:

```bash
python examples/transcribe.py
```

You will be prompted to select an audio file from the `audio-test` folder and choose whether to save the transcript.

### Recording Audio
To record audio and transcribe it, use:

```bash
python examples/record.py
```

You will be prompted to select an input device and start recording.

## Directory Structure

├── examples
│ ├── recording_transcribe.py
│ └── transcribe.py
├── pyscript
│ ├── audio_processing.py
│ ├── audio_recording.py
│ ├── transcription.py
│ └── transcriptor.py
├── utils
│ └── mini_librispeech_prepare.py
├── .gitignore
├── requirements.txt
└── README.md