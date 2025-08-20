from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('fraud_detector_1.pkl', 'rb'))
app = Flask(__name__)

# This is the endpoint that receives data
@app.route('/', methods=['POST'])
def predict():
    print("Hello World!")
    amt= request.form.get('amount')
    time = request.form.get('time')
    date = request.form.get('date')
    platform = request.form.get('platform')
    transaction_type = request.form.get('type')

    # Create the input array
    input_query = np.array([[float(amt), int(time), int(date)]])

    # Make the prediction
    result = model.predict(input_query)

    return jsonify({'Fraud Possibility': str(result)})

if __name__ == "__main__":
    app.run()