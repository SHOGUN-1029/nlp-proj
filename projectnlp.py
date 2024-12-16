import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from langdetect import detect, LangDetectException
import pickle
import numpy as np
import re


# Caching functions for model and resource loading
@st.cache_resource
def load_translation_model(model_path):
    """Load the translation model and tokenizer"""
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


@st.cache_resource
def load_prediction_model():
    """Load the disease prediction model and label encoder"""
    # Load the trained model from the pickle file
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)


    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)

    return model, label_encoder


# Symptoms list
SYMPTOMS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
    'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
    'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
    'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
    'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
    'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
    'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
    'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
    'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]


def detect_language(text):
    """Detect the language of the input text"""
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return None


def translate_text(model, tokenizer, text):
    inputs = tokenizer([text], return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = model.generate(**inputs, max_length=128)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation


def map_input_to_features(user_input, symptoms):
    """Map symptoms from user input to feature vector"""
    features = np.zeros(len(symptoms))
    for idx, symptom in enumerate(symptoms):
        if re.search(r'\b' + re.escape(symptom) + r'\b', user_input.lower()):
            features[idx] = 1
    return features


def main():

    st.title("🩺 Multilingual Med Assist")



    hindi_model_path = "saved_translator_model"
    malayalam_model_path = "saved_translator_modell"


    try:
        hindi_model, hindi_tokenizer = load_translation_model(hindi_model_path)
        malayalam_model, malayalam_tokenizer = load_translation_model(malayalam_model_path)
        prediction_model, label_encoder = load_prediction_model()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return


    input_text = st.text_area("Enter Symptoms (in Hindi or Malayalam)",
                              height=200,
                              placeholder="Type your symptoms here...")

    if st.button("Predict Disease"):
        if not input_text:
            st.warning("⚠️ Please enter some symptoms.")
            return


        try:
            lang = detect_language(input_text)
            st.info(f"🌐 Language detected: {lang}")
        except Exception as e:
            st.error(f"Language detection error: {e}")
            return


        if lang == "hi":  # Hindi
            with st.spinner("Translating Hindi to English..."):
                translated_text = translate_text(hindi_model, hindi_tokenizer, input_text)
        elif lang == "ml":  # Malayalam
            with st.spinner("Translating Malayalam to English..."):
                translated_text = translate_text(malayalam_model, malayalam_tokenizer, input_text)
        else:
            st.warning(f"⚠️ Detected language: {lang}. Sorry, we only support Hindi and Malayalam.")
            return

        # Display translated text
        st.text_area("Translated Text", value=translated_text, height=100, disabled=True)

        # Predict disease from translated text
        try:
            # Map translated text to feature vector
            input_features = map_input_to_features(translated_text, SYMPTOMS)

            prediction = prediction_model.predict(input_features.reshape(1, -1))
            predicted_disease = label_encoder.inverse_transform(prediction)

            st.success(f"🩺 Predicted Disease: {predicted_disease[0]}")
        except Exception as e:
            st.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main()