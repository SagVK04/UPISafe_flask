from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('fraud_detector_1.pkl', 'rb'))
app = Flask(__name__)

# This is the endpoint that receives data
@app.route('/', methods=['POST'])
def predict():
    print("Hello World!")
    amount= request.form.get('amount')
    time = request.form.get('time')
    date = request.form.get('date')

    # Create the input array & Make the prediction
    result = model.predict_proba(np.array([[int(date), int(amount),int(time)]]))
    fraud_score = result[0][1]

    return jsonify({'fraud_score': str(int(fraud_score*100))})

if __name__ == "__main__":
    app.run()