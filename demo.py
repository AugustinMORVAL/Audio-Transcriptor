import gradio as gr
from pyscript import Transcriptor
import os

transcriptor = Transcriptor(model_size="small")

demo_dir = "audio-test"
demo_files = {
    "Short Sample": os.path.join(demo_dir, "harvard.wav"),
    "Noise Sample": os.path.join(demo_dir, "jackhammer.wav"),
    "Meeting Sample 1 person": os.path.join(demo_dir, "meeting-clip1.wav"),
    "Meeting Sample 2 people": os.path.join(demo_dir, "meeting-clip2.wav"),
}

def process_audio(audio_path, enhancement):
    if audio_path is None:
        raise ValueError("Please provide an audio file.")
    
    transcription = transcriptor.transcribe_audio(audio_path, enhanced=enhancement)
    return str(transcription)

def create_download(text):
    os.makedirs(".temp", exist_ok=True)
    temp_file = ".temp/transcription.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(text)
    return temp_file

interface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio"),
        gr.Radio(choices=[True, False], value=False, label="Audio Enhancement", info="Enable for noisy audio")
    ],
    outputs=gr.Textbox(
        label="Complete Transcription", 
        interactive=True,
        info="You can edit the transcription here"
    ),
    title="🎙️ Audio Transcription Tool",
    description="""
    ⚠️ **Performance Notice**: This application performs intensive computations that are optimized for GPU usage. 
    If running on CPU only, transcription may take significantly longer (5-10x slower). For the best experience, 
    using a system with GPU is recommended.
    
    Upload an audio file or record directly to get a transcription.
    """,
    examples=[
        [demo_files["Short Sample"], False],
        [demo_files["Noise Sample"], True],
        [demo_files["Meeting Sample 1 person"], False],
        [demo_files["Meeting Sample 2 people"], False],
    ],
    cache_examples=True,
    cache_mode="eager",
    allow_flagging="never"
)

with gr.Blocks() as demo:
    interface.render()
    with gr.Column():
        download_button = gr.Button("📥 Download Edited Transcription")
        file_output = gr.File(label="Download Transcription")
    
    textbox = interface.output_components[0]
    
    download_button.click(fn=create_download, inputs=[textbox], outputs=[file_output])

if __name__ == "__main__":
    demo.launch(share=False)
