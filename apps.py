import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from langdetect import detect, LangDetectException
import pickle
import numpy as np
import re

@st.cache_resource
def load_model(model_path):
    """Load the model and tokenizer"""
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def translate_text(model, tokenizer, text):
    """Translate text to English"""
    inputs = tokenizer([text], return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = model.generate(**inputs, max_length=128)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation


def detect_language(text):
    """Detect the language of the input text"""
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return None


def main():
    st.title("🔤 Multilingual Translator (Hindi/Malayalam to English)")
    st.write("Enter text below to detect the language and translate it to English.")

    # Paths to the models
    hindi_model_path = "saved_translator_model"  # Path to the Hindi-to-English model
    malayalam_model_path = "saved_translator_modell"  # Path to the Malayalam-to-English model

    # Load models
    hindi_model, hindi_tokenizer = load_model(hindi_model_path)
    malayalam_model, malayalam_tokenizer = load_model(malayalam_model_path)

    input_text = st.text_area("Enter Text", height=200, placeholder="Type text here...")
    if st.button("Translate"):
        if input_text:
            with st.spinner("Detecting language..."):
                lang = detect_language(input_text)

            if lang:
                st.success(f"🌐 Detected language: {lang}")
            else:
                st.error("❌ Unable to detect the language. Please try again with a different text.")
                return

            if lang == "hi":  # Hindi
                with st.spinner("Translating Hindi to English..."):
                    translation = translate_text(hindi_model, hindi_tokenizer, input_text)
                    st.text_area("English Translation", value=translation, height=200, disabled=True)
            elif lang == "ml":  # Malayalam
                with st.spinner("Translating Malayalam to English..."):
                    translation = translate_text(malayalam_model, malayalam_tokenizer, input_text)
                    st.text_area("English Translation", value=translation, height=200, disabled=True)
            else:
                st.warning(f"⚠️ Detected language: {lang}. Sorry, we don't have a model for this language yet. Come back later for updates!")
        else:
            st.warning("⚠️ Please enter some text to translate.")



if __name__ == "__main__":
    main()
