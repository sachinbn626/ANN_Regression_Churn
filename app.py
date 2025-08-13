import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

st.set_page_config(page_title="Customer Churn Prediction")

# --- Load model (be tolerant to DTypePolicy coming from tf.keras) ---
try:
    from tensorflow.keras.mixed_precision import Policy
    from tensorflow.keras.utils import custom_object_scope
    with custom_object_scope({'DTypePolicy': Policy}):
        model = tf.keras.models.load_model("model.h5", compile=False)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# --- Load encoders & scaler ---
with open('onehot_encoder_geo.pkl', 'rb') as f:
    ohe_geo: OneHotEncoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    le_gender: LabelEncoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler: StandardScaler = pickle.load(f)

# --- UI ---
st.title("Customer churn prediction")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
# Show choices for Geography from the fitted OneHotEncoder
try:
    geo_categories = list(ohe_geo.categories_[0])
except Exception:
    geo_categories = []
geography = st.selectbox("Geography", geo_categories if geo_categories else ["France", "Germany", "Spain"])

# Show choices for Gender from the fitted LabelEncoder
gender_choices = list(le_gender.classes_) if hasattr(le_gender, "classes_") else ["Male", "Female"]
gender = st.selectbox("Gender", gender_choices)

age = st.number_input("Age", min_value=18, max_value=100, value=45)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=120000.50, format="%.2f")
num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.00, format="%.2f")

# --- Build one row of features ---
# Numeric/base features (all numeric)
num_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [float(balance)],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [float(estimated_salary)],
})

# Label-encode Gender -> numeric column named 'Gender'
gender_encoded = le_gender.transform([gender])[0]
gender_df = pd.DataFrame({"Gender": [int(gender_encoded)]})

# One-hot encode Geography properly (2D input, not nested Series)
# NOTE: ohe_geo expects shape (n_samples, 1)
geo_encoded = ohe_geo.transform([[geography]]).toarray()
geo_cols = ohe_geo.get_feature_names_out(['Geography'])
geo_df = pd.DataFrame(geo_encoded, columns=geo_cols)

# Combine in a single row
X = pd.concat([num_df, gender_df, geo_df], axis=1)

# If the scaler was fitted with specific column order, align to it
if hasattr(scaler, "feature_names_in_"):
    missing = set(scaler.feature_names_in_) - set(X.columns)
    extra = set(X.columns) - set(scaler.feature_names_in_)
    if missing:
        st.error(f"Missing expected features for scaler: {sorted(missing)}")
        st.stop()
    # reorder and drop any extras
    X = X.reindex(columns=scaler.feature_names_in_)

# Scale
try:
    X_scaled = scaler.transform(X)
except Exception as e:
    st.error(f"Scaling failed: {e}\nColumns provided: {list(X.columns)}")
    st.stop()

# Predict
try:
    pred = model.predict(X_scaled)
    prob = float(pred[0][0])
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

st.write(f"**Churn probability:** {prob:.2f}")

if prob > 0.5:
    st.success("The customer is likely to churn.")
else:
    st.info("The customer is likely **not** to churn.")
