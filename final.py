import streamlit as st
import pickle
import numpy as np
import re

# Load the trained model from the pickle file
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)
    st.write("Model Loaded Successfully!")

# Load the LabelEncoder from a pickle file (or recreate it if saved during training)
label_encoder_filename = 'label_encoder.pkl'
with open(label_encoder_filename, 'rb') as file:
    label_encoder = pickle.load(file)
    st.write("Label Encoder Loaded Successfully!")

# List of all symptoms in your dataset (the feature columns)
symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
    'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
    'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
    'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
    'mild_fever',
    'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    'swelled_lymph_nodes',
    'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
    'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising',
    'obesity',
    'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties',
    'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
    'hip_joint_pain',
    'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance',
    'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
    'irritability',
    'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation',
    'dischromic_patches',
    'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections',
    'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload',
    'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
    'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
    'yellow_crust_ooze'
]

# Streamlit app title
st.title("Disease Prediction App")

# Input text area for symptoms
user_input = st.text_area("Enter the symptoms (comma separated or as a sentence):")


# Function to map symptoms from user input to feature vector
def map_input_to_features(user_input, symptoms):
    # Initialize a feature vector with all zeros
    features = np.zeros(len(symptoms))

    # Use regular expression to find symptoms in user input
    for idx, symptom in enumerate(symptoms):
        if re.search(r'\b' + re.escape(symptom) + r'\b', user_input.lower()):
            features[idx] = 1  # Mark the symptom as present

    return features


# When the user clicks the predict button
if st.button('Predict'):
    # Map the user input to features
    input_features = map_input_to_features(user_input, symptoms)

    # Make the prediction
    prediction = model.predict(input_features.reshape(1, -1))

    # Convert the predicted label back to the actual disease name
    predicted_disease = label_encoder.inverse_transform(prediction)

    # Display the result
    st.write("Predicted Disease:", predicted_disease[0])
