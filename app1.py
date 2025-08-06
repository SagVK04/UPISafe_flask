import streamlit as st
import pickle
import numpy as np

# Load the model
try:
    model = pickle.load(open('fraud_detector.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'fraud_detector.pkl' not found. Make sure the model file is in the same directory.")
    st.stop()

st.title('💳 Fraud Detector App')
st.markdown("Use the controls below to input transaction details and predict the likelihood of fraud.")

# Create input widgets
st.header("Transaction Details")

amount = st.number_input('Transaction Amount')
time = st.slider('Time of Day (24-hour format)', 0, 23, 12)
date = st.slider('Day of the Month', 1, 31, 15)
platform_map = {1: 'GPay', 2: 'PhonePe', 3: 'Paytm', 4: 'Bharat Pe', 5:'Paypal'}
platform_selection = st.selectbox('Transaction Platform', options=list(platform_map.keys()), format_func=lambda x: platform_map[x])
type_map = {1: 'Debit', 2: 'Credit'}
type_selection = st.selectbox('Transaction Type', options=list(type_map.keys()), format_func=lambda x: type_map[x])

# Create a button to trigger the prediction
if st.button('Predict Fraud Possibility'):
    # Prepare the input data for the model
    input_query = np.array([[float(amount), int(time), int(date), int(platform_selection), int(type_selection)]])

    # Make the prediction
    result = model.predict(input_query)[0]

    st.subheader("Prediction Result")
    if result == 1:
        st.error('🚨 This transaction is **likely fraudulent**.')
    else:
        st.success('✅ This transaction appears **legitimate**.')