import streamlit as st
import pandas as pd
import pickle as pickle
import os
from streamlit_lottie import st_lottie
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient


def label_encode_column(data, column):
    lb = LabelEncoder()
    data[column] = lb.fit_transform(data[column])
    return data, lb


def upload_image():
    document_files = st.file_uploader("Upload the Image of the Animal Here : ",
                                      type=["pdf", "docx", "png", "jpg", "jpeg", "webp"],
                                      accept_multiple_files=True)
    if document_files is not None:
        save_folder = "images"
        os.makedirs(save_folder, exist_ok=True)

        image_paths = []
        for i, file in enumerate(document_files):
            file_path = os.path.join(save_folder, f"image_{i + 1}.{file.type.split('/')[-1]}")
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            image_paths.append(file_path)

        for path in image_paths:
            st.image(path, caption='Image', width=300)

        return image_paths

    if st.button("Upload Image"):
        st.success("Image Uploaded Successfully")


def model():
    df = pd.read_csv("data/animal_disease_dataset.csv")
    df.loc[:, 'Animal'] = 'cow'
    column_to_encode = 'Disease'

    df, encoder = label_encode_column(df, column_to_encode)
    df = df.drop("Animal", axis=1)
    new_features = ['blisters on gums', 'blisters on hooves', 'blisters on mouth',
                    'blisters on tongue', 'chest discomfort', 'chills', 'crackling sound',
                    'depression', 'difficulty walking', 'fatigue', 'lameness', 'loss of appetite',
                    'painless lumps', 'shortness of breath', 'sores on gums', 'sores on hooves',
                    'sores on mouth', 'sores on tongue', 'sweats', 'swelling in abdomen',
                    'swelling in extremities', 'swelling in limb', 'swelling in muscle',
                    'swelling in neck']

    for feature in new_features:
        df[feature] = 0

    for index, row in df.iterrows():
        for symptom_column in ['Symptom 1', 'Symptom 2', 'Symptom 3']:
            symptom = row[symptom_column]
            if symptom in new_features:
                df.loc[index, symptom] = 1

    df.drop(['Symptom 1', 'Symptom 2', 'Symptom 3'], axis=1, inplace=True)

    # Make diseae the last column
    cols = list(df.columns)
    cols.remove('Disease')
    cols.append('Disease')
    df = df[cols]

    # train test split
    X = df.drop("Disease", axis=1)
    Y = df['Disease']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_test, X_train, y_test, y_train = train_test_split(X, Y, test_size=50, random_state=12)
    svc = SVC()
    svc.fit(X_train, y_train)
    pred = svc.predict(X_test)
    accuracy = accuracy_score(y_test, pred) * 100
    with open('svm_model.pkl', 'wb') as model_file:
        pickle.dump(svc, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    st.success("Model and Scaler saved successfully")


def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def display_animation():
    lottie_anim = load_lottie("animations/cow_animation.json")
    st_lottie(lottie_anim, speed=1, reverse=False, loop=True, height=200, width=200, quality="high")


def image_model(img):
    CLIENT = InferenceHTTPClient(
        api_url="https://classify.roboflow.com",
        api_key="93J2JA4QhOvsFFy3ENF1"
    )

    result = CLIENT.infer(img, model_id="cow-diseae-identifier/1")

    confidence = result.get("confidence")
    top = result.get("top")
    st.write(f"Prediction : {top} with a confidence of {confidence * 100}%")


def main():
    st.set_page_config(
        page_title="Livestock Disease Predictor",
        page_icon=":cow2:",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    animation_col, header_col = st.columns([1, 3])

    with header_col:
        st.markdown("<div style='padding-top: 50px;'></div>", unsafe_allow_html=True)
        st.header("CATTLE DISEASE PREDICTOR")

    with animation_col:
        display_animation()

    with open('cattle_diseases_svc.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open('livestock_scaler.pkl', 'rb') as model_file:
        loaded_scaler = pickle.load(model_file)

    age = st.slider('What is the Age of the Animal in Years?', 0, 30)

    gender = st.selectbox("What is the Gender of the Animal?", ["Male", "Female"])
    temp = st.text_input("What is the Temperature of the Animal in Degrees Celsius?")
    if temp:
        temp = float(temp)
        temp = (temp * 9 / 5) + 32

    weight = st.slider("What is the Weight of the Animal in KG?", 0, 500)
    vaccination_history = st.selectbox("Has the Animal been previously Vaccinated?", ["Yes", "No"])
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
        # st.dataframe(new_data)

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

        new_df = loaded_scaler.transform(new_df)
        svc_model = loaded_model
        prediction = svc_model.predict(new_df)
        disease_names = {0: "Anthrax", 1: "Blackleg", 2: "Foot and Mouth", 3: "Lumpy Virus", 4: "Pneumonia"}
        predicted_disease = disease_names[prediction[0]]

        st.write(f"Prediction: {predicted_disease}")
        probabilities = svc_model.predict_proba(new_df)[0] * 100
        diseases = ["Anthrax", "Blackleg", "Foot and Mouth", "Lumpy Virus", "Pneumonia"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(diseases, probabilities)
        ax.set_xlabel('Disease')
        ax.set_ylabel('Probability (%)')
        ax.set_title('Probabilities of Different Diseases')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    img = upload_image()
    if img:
        with st.spinner("Processing Image..."):
            image_model(img)


if __name__ == "__main__":
    main()
