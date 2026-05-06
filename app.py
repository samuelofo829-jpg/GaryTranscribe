import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="GaryTranscribe Stable", page_icon="🎙️")

st.title("🎙️ GaryTranscribe Stable")

# Stable Model Choice
@st.cache_resource
def load_model():
    return whisper.load_model("base")

try:
    model = load_model()
except Exception as e:
    st.error("Model loading failed. Try clicking 'Clear Cache' in the sidebar.")

uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None:
    # We use a copy of the bytes to keep the player stable
    file_details = uploaded_file.getvalue()
    st.audio(file_details)
    
    if st.button("Start Transcription"):
        with st.spinner("Processing..."):
            # Use a simpler temp file approach
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(file_details)
            
            try:
                # Run transcription
                result = model.transcribe(tfile.name)
                text = result["text"].strip()
                
                st.subheader("Result:")
                st.info(text)
                
                st.download_button("Download Text", text, file_name="transcript.txt")
            except Exception as e:
                st.error(f"Transcription Error: {e}")
            finally:
                tfile.close()
                os.unlink(tfile.name)

# Sidebar helper
with st.sidebar:
    if st.button("Clear Cache & Reset"):
        st.cache_resource.clear()
        st.rerun()
