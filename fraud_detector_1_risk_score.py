import pandas as pd
import pickle
from sklearn import linear_model

df = pd.read_csv("final_transactions.csv")
input1 = df.drop(columns=["Fraud Possibility"])
target1 = df['Fraud Possibility']
print(input1.head())

input_train = input1[-4800:]
input_test = input1[:-1200]

target1_train = target1[-4800:]
target1_test = target1[:-1200]

model1 = linear_model.LinearRegression()
model1.fit(input_train,target1_train)
target1_pred = model1.predict(input_test)
for pred in target1_pred:
    print(f"{pred * 100:.2f}%") # Format to two decimal places
print("-" * 30)
res = model1.predict([[ 251467,1082025,2132]])
print(f"Risk Score: {(res*100)[0]:.2f}%")

#print(f"Accuracy: {accuracy_score(Y_test,Y_pred)*100:.2f} %")

