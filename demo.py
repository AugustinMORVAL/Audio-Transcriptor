import gradio as gr
from pyscript import Transcriptor

demo = gr.Blocks()
transcriptor = Transcriptor(model_size="large-v3-turbo")

microphone_transcribe = gr.Interface(
    fn=transcriptor.transcribe_audio,
    inputs=[
        gr.Audio(sources="microphone", type="filepath", label="Microphone"),
        gr.Radio([True, False], value=True, label="Enable audio enhancement"),
    ],
    outputs=[
        gr.Textbox(label="Transcription"),
        # gr.File(label="Download Transcription"),
        # gr.Textbox(label="Console Output", lines=10)
    ],
    title="Audio-Transcription leveraging Whisper Model",
    description=(
        "Transcribe microphone recording or audio inputs and return the transcription with speaker diarization."
    ),
    allow_flagging="never",
)

file_transcribe = gr.Interface(
    fn=transcriptor.transcribe_audio,
    inputs=[
        gr.Audio(sources="upload", type="filepath", label="Audio file"),
        gr.Radio([True, False], value=True, label="Enable audio enhancement"),
    ],
    outputs=[
        gr.Textbox(label="Transcription"),
        # gr.File(label="Download Transcription"),
        # gr.Textbox(label="Console Output", lines=10)
    ],
    title="Audio-Transcription leveraging Whisper Model",
    description=(
        "Transcribe microphone recording or audio inputs and return the transcription with speaker diarization."
    ),
    allow_flagging="never",
)


with demo:
    gr.TabbedInterface([microphone_transcribe, file_transcribe], ["Microphone", "Audio file"])

demo.queue().launch()
