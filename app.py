import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap

# ====== Load scaler + model ======
with open("models/parkinson_classification_pipeline.pkl", "rb") as f:
    scaler, model = pickle.load(f)

# ====== Load sample data ======
@st.cache_data
def load_sample_data():
    return pd.read_csv("Parkinsson disease.csv")

sample_data = load_sample_data()

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

st.write("Please enter the following voice measurement values:")

# ====== User Input ======
user_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    user_data.append(value)

user_data = np.array(user_data).reshape(1, -1)

# ====== Prediction ======
if st.button("Predict Parkinson's"):
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)[0]
    prediction_proba = model.predict_proba(user_data_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 Parkinson's Disease Detected (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ No Parkinson's Disease Detected (Probability: {prediction_proba:.2f})")

    # Explainability
    st.subheader("📊 Feature Importance (SHAP Values)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(user_data_scaled)
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", ascending=False)

    st.dataframe(shap_df)
    st.write("Feature importance (SHAP Bar Plot):")
    shap.plots.bar(shap_values)

# ====== Random Example Data Prediction ======
st.subheader("🎲 Test with Random Example Data")

if st.button("Predict with Random Sample"):
    random_sample = sample_data.sample(1)
    st.write("Selected Example Data:")
    st.dataframe(random_sample)

    random_features = random_sample[feature_names].values.reshape(1, -1)
    random_scaled = scaler.transform(random_features)
    prediction = model.predict(random_scaled)[0]
    prediction_proba = model.predict_proba(random_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 Parkinson's Disease Detected (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ No Parkinson's Disease Detected (Probability: {prediction_proba:.2f})")

    # Explainability
    st.subheader("📊 Feature Importance (SHAP Values)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(random_scaled)
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", ascending=False)

    st.dataframe(shap_df)
    st.write("Feature importance (SHAP Bar Plot):")
    shap.plots.bar(shap_values)

