from flask import Flask,request,jsonify
import pickle
import numpy as np
import streamlit as st

model = pickle.load(open('fraud_detector.pkl','rb'))
app = Flask(__name__)

@app.route('/',methods=['POST'])
def predict():
    st.header("Hello!")
    amount = request.form.get('amount')
    time = request.form.get('time')
    date = request.form.get('date')
    platform = request.form.get('platform')
    type = request.form.get('type')


    input_query = np.array([[float(amount),int(time),int(date),int(platform),int(type)]])

    result = model.predict(input_query)[0]

    return jsonify({'Fraud Possibility': str(result)})
