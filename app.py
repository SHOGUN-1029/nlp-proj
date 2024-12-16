import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
import os


def verify_model_files(model_path):
    """Verify that all necessary model files exist"""
    required_files = ['config.json', 'tokenizer_config.json']

    if not os.path.exists(model_path):
        return False, f"Model directory '{model_path}' does not exist"

    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        return False, f"Missing required files: {', '.join(missing_files)}"

    return True, "All files present"


@st.cache_resource
def load_cached_model(model_path):
    """Load and cache the model and tokenizer"""
    try:
        # Verify model files exist
        files_exist, message = verify_model_files(model_path)
        if not files_exist:
            st.error(f"Model files verification failed: {message}")
            return None, None

        # Get absolute path
        abs_model_path = os.path.abspath(model_path)
        st.info(f"Loading model from: {abs_model_path}")

        # Load model and tokenizer
        model = TFAutoModelForSeq2SeqLM.from_pretrained(abs_model_path)
        tokenizer = AutoTokenizer.from_pretrained(abs_model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


class TranslatorApp:
    def __init__(self):
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, "saved_translator_model")
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer using cached function"""
        return load_cached_model(self.model_path)

    def translate(self, text):
        """Translate Hindi text to English"""
        try:
            # Tokenize
            inputs = self.tokenizer([text], return_tensors="tf", padding=True, truncation=True, max_length=128)

            # Generate translation
            outputs = self.model.generate(**inputs, max_length=128)

            # Decode
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
        except Exception as e:
            return f"Translation error: {str(e)}"


def main():
    st.set_page_config(
        page_title="Hindi to English Translator",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stTextArea textarea {
            font-size: 16px;
        }
        .stButton button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize app
    app = TranslatorApp()

    # Header
    st.title("🔤 Hindi to English Translator")
    st.write("Enter Hindi text below to get its English translation")

    # Show model path
    st.info(f"Model path: {app.model_path}")

    # Check model files before loading
    files_exist, message = verify_model_files(app.model_path)
    if not files_exist:
        st.error(f"❌ {message}")
        st.error("Please ensure the model files are in the correct location.")
        st.write("Expected directory structure:")
        st.code("""
saved_translator_model/
├── config.json
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.json
├── merges.txt
└── tf_model.h5 (or similar model file)
        """)
        return

    # Load model
    with st.spinner("Loading model..."):
        app.model, app.tokenizer = app.load_model()

    if app.model is None or app.tokenizer is None:
        st.error("❌ Failed to load model. Please check the error messages above.")
        return

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hindi Text")
        input_text = st.text_area(
            "",
            height=200,
            placeholder="Enter Hindi text here...",
            key="input_hindi"
        )

        if st.button("🔄 Translate", key="translate_button"):
            if input_text:
                with st.spinner("Translating..."):
                    translation = app.translate(input_text)
                    st.session_state.translation = translation
            else:
                st.warning("⚠️ Please enter some text to translate.")

    with col2:
        st.subheader("English Translation")
        if 'translation' in st.session_state:
            st.text_area(
                "",
                value=st.session_state.translation,
                height=200,
                disabled=True,
                key="output_english"
            )
        else:
            st.text_area(
                "",
                value="Translation will appear here...",
                height=200,
                disabled=True,
                key="output_placeholder"
            )



if __name__ == "__main__":
    main()