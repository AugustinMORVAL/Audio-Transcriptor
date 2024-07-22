import os
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
import base64
from pyannote.audio import Pipeline
import time
from pydub import AudioSegment
import tempfile

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

client = Groq(api_key=GROQ_API_KEY)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)


def transcript_audio(file, segment=None):
    with st.spinner('Writting transcript...'):
        if detect_speaker == "On":
            transcriptions = []
            for turn, _, speaker in segments:
                audio = AudioSegment.from_file(file)
                segment = audio[turn.start*1000:turn.end*1000]
                segmented_file = segment.export(format="wav")
                transcription = client.audio.transcriptions.create(
                    file = ("file.wav", segmented_file.read()),
                    model = "whisper-large-v3"
                )
                transcriptions.append((speaker, transcription.text))
            return "\n".join([f"{speaker}:\n {text}" for speaker, text in transcriptions])
        else:
            transcription = client.audio.transcriptions.create(
                file = file,
                model = "whisper-large-v3"
            )
            return transcription.text


def diarization(uploaded_file):
    try:
        with st.spinner('Processing audio file...'):
            start_time = time.time()
            diarization = pipeline(uploaded_file)
            end_time = time.time()
            st.success(f'Audio file processed successfully in {end_time - start_time} seconds.')
    except Exception as e:
        st.error(f'Failed to process the audio file: {e}')
    segments = list(diarization.itertracks(yield_label=True))
    return segments

def file_object_to_path(file_object):
    # Create a temporary file
    temp = tempfile.NamedTemporaryFile(delete=False)
    # Write the contents of the file object to the temporary file
    temp.write(file_object.read())
    # Close the temporary file
    temp.close()
    # Return the path to the temporary file
    return temp.name


st.title("Audio Transcription")
st.markdown("Upload an audio file and click the **Transcript** button to see the transcription.")

# Options section
detect_speaker = st.radio("Speaker Detection", options=["On", "Off"], index=0)

uploaded_file = st.file_uploader("Choose an audio file", type=["WAV", "MP3", "MP4", "M4A", "OGG", "OGA", "FLAC", "AAC", "WMA", "AMR"])

if uploaded_file is not None:
    if st.button("Transcript"):
        if detect_speaker == "On":
            segments = diarization(uploaded_file)
            st.write(segments)
        transcript = transcript_audio(uploaded_file, segments if detect_speaker == "On" else None)
        st.write(transcript)

        # # Optional: Download the transcript as a text file
        # b64 = base64.b64encode(transcript.encode()).decode()
        # href = f'<a href="data:file/txt;base64,{b64}" download="transcript.txt">Download Transcript</a>'
        # st.markdown(href, unsafe_allow_html=True)


