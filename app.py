import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model_filename = "fair_price_model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Currency Conversion Rate (NZD to INR)
NZD_TO_INR = 50  # Update this with real-time data if needed

# Ensure feature order matches training
expected_features = ["Cost_per_Unit", "Demand_Score", "Imports Qty", "Seasonality_Factor", "Competitor_Price"]

# Streamlit UI
st.title("🛍️ Fair Price Prediction for Artisan Products (₹ INR)")

st.sidebar.header("Enter Product Details")

# Input Fields
cost_per_unit = st.sidebar.number_input("Cost Per Unit (in INR)", min_value=0.0, step=0.1, format="%.2f")
demand_score = st.sidebar.slider("Demand Score (1.0 - 2.0)", 1.0, 2.0, step=0.1)
imports_qty = st.sidebar.number_input("Imports Quantity", min_value=1, step=1)
seasonality_factor = st.sidebar.slider("Seasonality Factor (1.0 - 1.5)", 1.0, 1.5, step=0.1)
competitor_price = st.sidebar.number_input("Competitor Price (in INR)", min_value=0.0, step=0.1, format="%.2f")

# Predict Fair Price
if st.sidebar.button("Predict Fair Price"):
    # Ensure input matches trained model's expected feature order
    input_data = pd.DataFrame([[cost_per_unit, demand_score, imports_qty, seasonality_factor, competitor_price]], 
                              columns=expected_features)

    # Make prediction (already in INR)
    predicted_price_inr = model.predict(input_data)[0]

    st.success(f"💰 Recommended Fair Price: ₹{predicted_price_inr:.2f} INR")

st.write("This tool helps artisans set a fair price for their products based on cost and demand trends.")
