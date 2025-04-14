
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Load models
xgb_model = joblib.load("model_xgb.pkl")
lstm_model = load_model("model_lstm.h5", compile=False)
scaler = joblib.load("scaler_lstm.pkl")
sarimax_model = SARIMAXResults.load("sarimax_model.pkl")

# Correct feature list used during training
features = ['High', 'Low', 'Open', 'entry_count_ma', 'cumulative_negative',
            'close_20_ma', 'close_50_ma', 'close_lag_5', 'close_lag_7', 'rolling_mean_10d']

# App title
st.title("TREND-TRACKER")

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
        st.success(f"Predicted Close Price (XGBoost): {prediction:.2f}")

    elif model_choice == "LSTM":
        # Repeat input to simulate time window for LSTM
        sequence = pd.concat([input_df] * 60, ignore_index=True)

        # Add dummy 'Close' column to match scaler structure
        sequence['Close'] = 0.0

        scaled_seq = scaler.transform(sequence)
        X_lstm = scaled_seq[-60:, :-1].reshape(1, 60, -1)
        y_pred_scaled = lstm_model.predict(X_lstm)[0][0]

        # Inverse transform
        min_val = scaler.data_min_[-1]
        max_val = scaler.data_max_[-1]
        close_price = y_pred_scaled * (max_val - min_val) + min_val

        st.success(f" Predicted Close Price (LSTM): {close_price:.2f}")

    else:
        # SARIMAX with exogenous input
        sarimax_pred = sarimax_model.forecast(steps=1, exog=input_df)[0]
        st.success(f"Predicted Close Price (SARIMAX): {sarimax_pred:.2f}")
