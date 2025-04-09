
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Load models
xgb_model = joblib.load("model_xgb.pkl")
lstm_model = load_model("model_lstm.h5")
scaler = joblib.load("scaler_lstm.pkl")
sarimax_model = SARIMAXResults.load("sarimax_model.pkl")

# Feature list (must match training)
features = ['High', 'Low', 'close_lag_1', 'Volume', 'entry_count',
            'cumulative_negative', 'positive_price_ratio',
            'cumulative_neutral', 'cumulative_positive', 'close_lag_10']

# App title
st.title("📈 NAB.AX Stock Close Price Predictor")

# Input form
st.subheader("Enter feature values below:")
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

input_df = pd.DataFrame([user_input])

# Model selection
model_choice = st.selectbox("Choose model to use:", ["XGBoost", "LSTM", "SARIMAX"])

if st.button("Predict"):
    if model_choice == "XGBoost":
        prediction = xgb_model.predict(input_df)[0]
        st.success(f"📊 Predicted Close Price (XGBoost): {prediction:.2f}")
    
    elif model_choice == "LSTM":
        # Create dummy sequence (repeat input to match LSTM window)
        sequence = pd.concat([input_df] * 60, ignore_index=True)
        scaled_seq = scaler.transform(sequence)
        X_lstm = scaled_seq[-60:, :-1].reshape(1, 60, -1)
        y_pred_scaled = lstm_model.predict(X_lstm)[0][0]

        # Inverse transform
        min_val = scaler.data_min_[-1]
        max_val = scaler.data_max_[-1]
        close_price = y_pred_scaled * (max_val - min_val) + min_val

        st.success(f"📊 Predicted Close Price (LSTM): {close_price:.2f}")
    
    else:
        # SARIMAX requires exogenous input
        sarimax_pred = sarimax_model.forecast(steps=1, exog=input_df)[0]
        st.success(f"📊 Predicted Close Price (SARIMAX): {sarimax_pred:.2f}")
