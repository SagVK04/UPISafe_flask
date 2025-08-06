import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('final_cleaned_upi_transactions_reordered.csv')

#KNN Classifiers need all feature values in digits

#print(df.shape)
#print(df.sample(5))
X = df.drop(columns=['Fraud Possibility'])

Y = df['Fraud Possibility']

X_train = X[:-1400]
X_test = X[-1340:]
#Standardizing the features(X) to increase accuracy
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.fit_transform(X_test)

Y_train = Y[:-1400]
Y_test = Y[-1340:]

fraud_model = KNeighborsClassifier(n_neighbors=1113)
fraud_model.fit(X_train_scaled, Y_train)
Y_pred = fraud_model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(Y_test,Y_pred)*100:.2f} %")


import pickle
pickle.dump(fraud_model,open('fraud_detector.pkl','wb'))