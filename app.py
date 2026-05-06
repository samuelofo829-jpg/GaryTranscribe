import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="ArchæoTranscribe", page_icon="⛏️")

st.title("⛏️ ArchæoTranscribe")
st.caption("Specialized for Excavation & Field Recording")

@st.cache_resource
def load_model():
    # 'base.en' is tuned specifically for English technical terms
    return whisper.load_model("base.en")

model = load_model()

# This is the secret sauce: Feeding the AI technical context
ARCHAEO_CONTEXT = (
    "archaeology, site excavation, datum line, elevation, zero point, "
    "hand trowels, dustpans, 10 cm increments, arbitrary levels, "
    "labeling protocols, unit and level numbers, vertical control."
)

uploaded_file = st.file_uploader("Upload Dig Site Audio", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None:
    audio_bytes = uploaded_file.getvalue()
    st.audio(audio_bytes)
    
    if st.button("Analyze Excavation Audio"):
        with st.spinner("Applying Archaeology Language Model..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            try:
                # We use the 'initial_prompt' to force the AI to recognize your specific terms
                result = model.transcribe(
                    tmp_path, 
                    initial_prompt=ARCHAEO_CONTEXT
                )
                
                st.subheader("Field Transcript:")
                st.markdown(f"```\n{result['text'].strip()}\n```")
                
                st.download_button(
                    "Download Field Notes", 
                    result["text"], 
                    file_name=f"excavation_unit_{uploaded_file.name}.txt"
                )
            except Exception as e:
                st.error(f"Field Error: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

with st.sidebar:
    st.info("**Context Active:** AI is currently tuned for: Datum Lines, Trowels, and 10cm Arbitrary Levels.")
