import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Room Occupancy", page_icon="🏠")
page_bg = """
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1497366754035-f200968a6e72");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
    right: 2rem;
}

.block-container {
    background-color: rgba(255,255,255,0.85);
    padding: 2rem;
    border-radius: 15px;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

st.title("🏠 Room Occupancy Detection")

st.write("Predict whether the room is occupied or not.")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.header("Enter Room Sensor Values")

temperature = st.number_input("Temperature", value=23.0)
humidity = st.number_input("Humidity", value=27.0)
light = st.number_input("Light", value=300.0)
co2 = st.number_input("CO2", value=700.0)
humidity_ratio = st.number_input("Humidity Ratio", value=0.004)

if st.button("Predict"):

    data = pd.DataFrame(
        [[temperature, humidity, light, co2, humidity_ratio]],
        columns=["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    )

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        st.success("✅ Room is Occupied")
    else:
        st.error("❌ Room is Unoccupied")
