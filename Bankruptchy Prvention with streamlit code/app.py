import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Bankruptcy Prediction", layout="centered")

st.title("🏦 Bankruptcy Prediction App")

# ----------------------------------------------------
# Load model safely
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("bankruptcy_best_model.pkl")

model = load_model()

st.success("✅ Model Loaded Successfully")

# ----------------------------------------------------
# Try to detect feature names automatically
# ----------------------------------------------------
def get_feature_names(model):
    # Case 1: Model saved inside pipeline
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    # Case 2: Normal sklearn/xgboost model
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return None

FEATURES = get_feature_names(model)

# ----------------------------------------------------
# If feature names not found, allow CSV-based prediction
# ----------------------------------------------------
if FEATURES is None:
    st.warning("⚠️ Feature names not found inside model. Use CSV upload instead.")
else:
    st.info(f"✅ Detected {len(FEATURES)} input features automatically.")

# ----------------------------------------------------
# User Input Mode Selection
# ----------------------------------------------------
mode = st.radio("Choose Input Method", ["Manual Input", "Upload CSV"])

# ----------------------------------------------------
# Manual input UI
# ----------------------------------------------------
if mode == "Manual Input" and FEATURES:

    st.subheader("📊 Enter Input Values")

    user_data = {}
    for col in FEATURES:
        user_data[col] = st.number_input(
            f"{col}",
            value=0.0,
            format="%.4f"
        )

    input_df = pd.DataFrame([user_data])

    st.write("### Your Input Data")
    st.dataframe(input_df)

    if st.button("Predict"):
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.subheader("📢 Prediction Result")
            if prediction == 1:
                st.error(f"⚠️ High Bankruptcy Risk\nProbability: {probability:.2f}")
            else:
                st.success(f"✅ Low Bankruptcy Risk\nProbability: {probability:.2f}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ----------------------------------------------------
# CSV Upload Mode
# ----------------------------------------------------
elif mode == "Upload CSV":

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(data)

        if st.button("Predict from CSV"):
            try:
                preds = model.predict(data)
                probs = model.predict_proba(data)[:, 1]

                data["Prediction"] = preds
                data["Probability"] = probs

                st.write("### Prediction Results")
                st.dataframe(data)

                # Download results
                csv = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results",
                    csv,
                    "bankruptcy_predictions.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
