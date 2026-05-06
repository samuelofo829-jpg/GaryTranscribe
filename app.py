import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="GaryTranscribe Pro", page_icon="🎙️")

st.title("🎙️ GaryTranscribe Pro")
st.markdown("High-accuracy transcription powered by Whisper-Small.")

# Load AI Model (Upgraded from 'base' to 'small')
@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None:
    # Use a persistent bytes object to prevent player errors
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format='audio/mp3')
    
    if st.button("Start Transcription"):
        with st.spinner("G is analyzing the audio... this 'Small' model is deeper, so give it a moment."):
            # Create a temp file correctly
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            try:
                # Transcribe
                result = model.transcribe(tmp_path)
                transcript_text = result["text"].strip()
                
                st.subheader("Transcription:")
                st.success("Complete!")
                st.write(transcript_text)
                
                # Fixed Download Button: We pass the text directly so no 'File Not Found'
                st.download_button(
                    label="Download Transcript (.txt)",
                    data=transcript_text,
                    file_name=f"transcript_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
