import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="GaryTranscribe", page_icon="🎙️")

st.title("🎙️ GaryTranscribe")
st.markdown("Upload any audio file to turn it into text for free.")

# Load AI Model (Base is balanced for speed and accuracy)
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    
    if st.button("Start Transcription"):
        with st.spinner("G is processing your audio... hang tight."):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Transcribe
            result = model.transcribe(tmp_path)
            
            st.subheader("Transcription:")
            st.success("Analysis complete!")
            st.write(result["text"])
            
            # Download option
            st.download_button("Download Transcript (.txt)", result["text"], file_name="transcript.txt")
            
            # Clean up the temp file
            os.remove(tmp_path)
