import joblib 
import numpy as np
import streamlit as st
# load karo training model ko 
model = joblib.load("housing_model.pkl")

st.set_page_config(page_title="Housing Price Predictor",page_icon="++++")
st.title("Clifornia  Housing Price Predictor")
st.write("Enter property and location details to predict the price of the house")

med_inc = st.number_input("Median income (in tens of thousands of dollars)", min_value=0.0, value=3.87)
house_age = st.number_input("Average house age (years)", min_value=0.0, value=28.0)
ave_rooms = st.number_input("Average rooms per household", min_value=0.0, value=5.43)
ave_bedrms = st.number_input("Average bedrooms per household", min_value=0.0, value=1.10)
population = st.number_input("Block population", min_value=0.0, value=1425.0)
ave_occup = st.number_input("Average occupants per household", min_value=0.0, value=3.07)
latitude = st.number_input("Latitude", value=34.05, format="%.4f")
longitude = st.number_input("Longitude", value=-118.25, format="%.4f")

if st.button("Predict House Price"):
    # Keep the inputs in the exact order used to train the model
    input_data = np.array([[
        med_inc,
        house_age,
        ave_rooms,
        ave_bedrms,
        population,
        ave_occup,
        latitude,
        longitude
    ]])

    prediction = model.predict(input_data)[0]

    # Dataset target values are measured in $100,000 units
    price_in_dollars = prediction * 100_000

    st.success(f"Estimated median house value: ${price_in_dollars:,.0f}")