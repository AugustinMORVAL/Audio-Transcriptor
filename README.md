# Audio Transcription and Diarization Tool

## Overview

This project provides a robust set of tools for transcribing audio files using the Whisper model and performing speaker diarization with PyAnnote. Users can process audio files, record audio, and save transcriptions with speaker identification.

# Audio Transcription and Diarization Tool

## Overview

This project provides a robust set of tools for transcribing audio files using the Whisper model and performing speaker diarization with PyAnnote. Users can process audio files, record audio, and save transcriptions with speaker identification.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Key Components](#key-components)
  - [Transcriptor](#transcriptor)
    - [Model Management](#model-management)
    - [Key Features](#key-features)
    - [Transcription Pipeline](#transcription-pipeline)
  - [AudioProcessor](#audioprocessor)
    - [Audio File Management](#audio-file-management)
    - [Enhancement Features](#enhancement-features)
    - [Parameter Optimization](#parameter-optimization)
    - [Audio Processing Pipeline](#audio-processing-pipeline)
- [Performance Considerations](#performance-considerations)
- [Usage](#usage)
  - [Basic Example](#basic-example)
- [Try the demo](#try-the-demo)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Features

- **Transcription**: Convert audio files in various formats to text (automatically converts to WAV).
- **Speaker Diarization**: Identify different speakers in the audio.
- **Speaker Retrieval**: Name speakers during transcription.
- **Audio Recording**: Record audio directly from a microphone.
- **Audio Preprocessing**: Includes resampling, format conversion, and audio enhancement.
- **Multiple Model Support**: Choose from various Whisper model sizes.

### Supported Whisper Models

This tool supports various Whisper model sizes, allowing you to balance accuracy and computational resources:

- **`tiny`**: Fastest, lowest accuracy
- **`base`**: Fast, good accuracy
- **`small`**: Balanced speed and accuracy
- **`medium`**: High accuracy, slower
- **`large`**: High accuracy, resource-intensive
- **`large-v3`**: Latest and most accurate
- **`large-v3-turbo`**: Optimized for faster processing

Specify the model size when initializing the Transcriptor

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

## Key Components

### Transcriptor

The `Transcriptor` class is the core engine for audio transcription and speaker diarization:

- **Model Management**:
  - Supports multiple Whisper model sizes (tiny to large-v3)
  - Automatic GPU detection and optimization
  - Efficient batch processing for GPU acceleration

- **Key Features**:
  - Automatic device selection (CPU/GPU)
  - Batched processing for memory efficiency
  - Progress tracking with detailed logging
  - Memory management for large files

#### Transcription Pipeline:

```
Input Audio File
      │
      ▼
┌────────────────┐
│ Audio          │
│ Preprocessing  │──┐
└────────────────┘  │
      │             │
      ▼             │
┌────────────────┐  │
│ Speaker        │  │
│ Diarization    │  │
└────────────────┘  │
      │             │
      ▼             │
┌────────────────┐  │
│ Audio          │  │
│ Segmentation   │  │
└────────────────┘  │
      │             │
      ▼             │
┌────────────────┐  │
│ Whisper Model  │  │
│ Transcription  │◄─┘
└────────────────┘
      │
      ▼
┌────────────────┐
│ Speaker        │
│ Identification │
│ (Optional)     │
└────────────────┘
      │
      ▼
Final Transcript
```

### AudioProcessor 

The `AudioProcessor` class handles all audio file preprocessing and enhancement:

- **Audio File Management**:
  - Format conversion (to WAV)
  - Sample rate adjustment (to 16kHz)
  - Duration and format tracking
  - Detailed change logging

- **Enhancement Features**:
  - Noise reduction with adjustable strength
  - Voice clarity enhancement
  - Volume normalization
  - Automatic parameter optimization
  - Spectral contrast improvement

- **Parameter Optimization**:
  - Grid search for optimal enhancement settings
  - Correlation analysis
  - Spectral contrast evaluation
  - Progress tracking during optimization

#### Audio Processing Pipeline:

```
Input Audio
    │
    ▼
┌─────────────┐
│ Format      │
│ Detection   │
└─────────────┘
    │
    ▼
┌─────────────┐     ┌─────────────┐
│ Format      │     │ Sample Rate │
│ Conversion  │────►│ Adjustment  │
└─────────────┘     └─────────────┘
                         │
                         ▼
┌───────────────────────────────────┐
│        Enhancement Pipeline       │
├───────────────────────────────────┤
│                                   │
│  ┌─────────────┐   ┌──────────┐   │
│  │   Spectral  │   │  Noise   │   │
│  │  Analysis   │──►│ Reduction│   │
│  └─────────────┘   └──────────┘   │
│         │              │          │
│         ▼              ▼          │
│  ┌─────────────┐   ┌──────────┐   │
│  │   Voice     │   │ Volume   │   │
│  │ Enhancement │◄──│ Normalize│   │
│  └─────────────┘   └──────────┘   │
│         │                         │
└─────────┼─────────────────────────┘
          │
          ▼
    Enhanced Audio
```

## Performance Considerations

- **Batch Processing**: Implements efficient batching for GPU processing to handle long audio files
- **Memory Optimization**: Includes automatic memory management and cleanup during processing
- **Enhancement Optimization**: Uses grid search with correlation analysis to find optimal enhancement parameters

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

## Try the demo

The project includes a user-friendly web interface built with Gradio for easy testing and demonstration. Check it out:

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- OpenAI for the Whisper model
- PyAnnote for the speaker diarization pipeline
- All contributors and users of this project