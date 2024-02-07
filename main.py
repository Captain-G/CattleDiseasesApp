import streamlit as st
import pandas as pd
import pickle as pickle
import numpy as np


def main():
    st.set_page_config(
        page_title="Livestock Disease Predictor",
        page_icon=":cow2:",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    st.header("Livestock Disease Predictor")
    age = st.slider('What is the Age of the Cow in Years?', 0, 30)
    temp = st.text_input("What is the Temperature of the Cow in Degrees Celsius?")
    if temp:
        temp = float(temp)
        temp = (temp * 9 / 5) + 32
    st.write("Select ONE Symptom Per Row : ")
    col1, col2, col3 = st.columns(3)
    with col1:
        symptom1 = st.radio("Select the First Symptom : ",
                            ["Depression", "Loss of Appetite", "Painless Lumps", "Difficulty Walking", "Lameness",
                             "Chills",
                             "Crackling Sound", "Shortness of Breath", "Chest Discomfort", "Fatigue"])
        symptom1 = symptom1.lower()
    with col2:
        symptom2 = st.radio("Select the Second Symptom : ",
                            ["Painless Lumps", "Swelling in Limbs", "Loss of Appetite", "Blisters on Gums",
                             "Depression", "Blisters on Tongue", "Blisters on Mouth", "Swellings in Extremities",
                             "Sores on Mouth", "Lameness"])
        symptom2 = symptom2.lower()
    with col3:
        symptom3 = st.radio("Enter the Third Symptom :",
                            ["Loss of Appetite", "Crackling Sound", "Depression", "Difficulty Walking",
                             "Painless Lumps", "Shortness of Breath", "Lameness", "Chills", "Swellings in Extremities",
                             "Fatigue"])
        symptom3 = symptom3.lower()
    if st.button("Show Prediction"):
        new_data = {'Age': f"{age}", 'Temperature': f"{temp}", 'Symptom 1': f'{symptom1}',
                    'Symptom 2': f'{symptom2}', 'Symptom 3': f'{symptom3}'}

        # Convert the new data to a DataFrame
        new_df = pd.DataFrame([new_data])

        new_features = ['blisters on gums', 'blisters on hooves', 'blisters on mouth',
                        'blisters on tongue', 'chest discomfort', 'chills', 'crackling sound',
                        'depression', 'difficulty walking', 'fatigue', 'lameness', 'loss of appetite',
                        'painless lumps', 'shortness of breath', 'sores on gums', 'sores on hooves',
                        'sores on mouth', 'sores on tongue', 'sweats', 'swelling in abdomen',
                        'swelling in extremities', 'swelling in limb', 'swelling in muscle',
                        'swelling in neck']
        # create columns
        for feature in new_features:
            new_df[feature] = 0

        for index, row in new_df.iterrows():
            for symptom_column in ['Symptom 1', 'Symptom 2', 'Symptom 3']:
                symptom = row[symptom_column]
                if symptom in new_features:
                    new_df.loc[index, symptom] = 1

        new_df.drop(['Symptom 1', 'Symptom 2', 'Symptom 3'], axis=1, inplace=True)
        model = pickle.load(
            open("/home/gachuki/PycharmProjects/LivestockDiseasesApp/model/cattle_diseases_model.pkl", "rb"))
        # scaler = pickle.load(
        #     open("/home/gachuki/PycharmProjects/LivestockDiseasesApp/model/cattle_diseases_scaler.pkl", "rb"))
        # scaler.transform(new_df)
        prediction = model.predict(new_df)
        if prediction == [0]:
            st.write(f"Prediction : Anthrax")
        elif prediction == [1]:
            st.write(f"Prediction : Blackleg")
        elif prediction == [2]:
            st.write(f"Prediction : Foot and mouth")
        elif prediction == [3]:
            st.write(f"Prediction : Lumpy virus")
        elif prediction == [4]:
            st.write(f"Prediction : Pneumonia")


if __name__ == "__main__":
    main()
