import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import accuracy_score

df = pd.read_csv('MyTransaction_updated.csv')
input_data = df.drop(columns=["Fraud Probability","Date.1","Category","RefNo"])
target = df['Fraud Probability']
input_data['Time'] = input_data['Time'].str.replace(':','').astype(int)
input_data['Date'] = input_data['Date'].str.replace('/','').astype(int)
input_data['Amount'] = input_data['Amount'].astype(int)
print(input_data.head())


input_train = input_data.iloc[:4800] #first 4800 elements
input_test = input_data.iloc[-1200:] #last 1200 elements

target_train = target.iloc[:4800]
target_test = target.iloc[-1200:]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(input_train,target_train)


prediction = model.predict_proba([[582017,5656,630]])
pred_res = model.predict([[582017,5656,630]])
if pred_res == 0:
    print(f"Predicted Result: Safe!")
if pred_res == 1:
    print(f"Predicted Result: Fraud!")

print(prediction[0])
fraud_pred = prediction[0][1]
print(f"Risk Score: {fraud_pred * 100:.2f}%")
print()

target_pred = model.predict(input_test)
print(f"Accuracy: {accuracy_score(target_test,target_pred)*100:.2f} %")
pickle.dump(model,open('fraud_detector_1.pkl','wb'))