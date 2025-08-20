import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('final_transactions.csv')
input = df.drop(columns=['Fraud Possibility'])
target = df['Fraud Possibility']
print(input)

input_train = input[-4800:] #first 4800 elements
input_test = input[:-1200] #last 1200 elements

target_train = target[-4800:]
target_test = target[:-1200]

model = tree.DecisionTreeClassifier()
model.fit(input,target)
print(model.predict([[ 547,3082025,1509]]))
res = model.predict([[ 547,3082025,1509]])
if(res == 0):
    print("Possibly Safe!")
else:
    print("Not Safe!")
target_pred = model.predict(input)
print(f"Accuracy: {accuracy_score(target,target_pred)*100:.2f} %")
pickle.dump(model,open('fraud_detector_1.pkl','wb'))