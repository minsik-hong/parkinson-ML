# app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap

# ====== Load scaler + model ======
with open("models/parkinson_classification_pipeline.pkl", "rb") as f:
    scaler, model = pickle.load(f)

# ====== Feature names ======
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
    'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# ====== App title ======
st.title("🧠 Parkinson's Disease Prediction App")

st.write("아래 음성 측정 데이터를 입력하세요.")
st.write("Please enter the following voice measurement values:")

# ====== User Input ======
user_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    user_data.append(value)

user_data = np.array(user_data).reshape(1, -1)

# ====== Prediction ======
if st.button("Predict Parkinson's"):
    # preprocess (Scaler)
    user_data_scaled = scaler.transform(user_data)

    # predict
    prediction = model.predict(user_data_scaled)[0]
    prediction_proba = model.predict_proba(user_data_scaled)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"🚨 Parkinson's Disease Detected (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ No Parkinson's Disease Detected (Probability: {prediction_proba:.2f})")

    # ====== SHAP Explainability ======
    st.subheader("📊 Feature Importance (SHAP Values)")

    # Prepare SHAP explainer based on model
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the input sample
    shap_values = explainer(user_data_scaled)

    # Organize SHAP values into DataFrame
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", ascending=False)

    # Display feature importance as DataFrame
    st.dataframe(shap_df)

    # Bar plot visualization
    st.write("Feature importance (SHAP Bar Plot):")
    shap.plots.bar(shap_values)

