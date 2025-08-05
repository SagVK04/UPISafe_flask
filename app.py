from flask import Flask,request,jsonify
import pickle
import numpy as np

model = pickle.load(open('fraud_detector.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict',methods=['POST'])
def predict():
    amount = request.form.get('amount')
    time = request.form.get('time')
    date = request.form.get('date')
    platform = request.form.get('platform')
    type = request.form.get('type')


    input_query = np.array([[float(amount),int(time),int(date),int(platform),int(type)]])

    result = model.predict(input_query)[0]

    return jsonify({'Fraud Possibility': str(result)})

if __name__ == '__main__':
    app.run(debug=True)