
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import os
# Get the list of all files and directories
dirname = os.path.dirname(__file__)
csv_path = os.path.join(dirname, '../resources/diabetes.csv')
df = pd.read_csv(csv_path)
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

print(X_test)

single_data = X_test.iloc[0].values.reshape(1, -1)
single_pred = model.predict(single_data)

prediction_label = "Yes" if single_pred[0] == 1 else "No"
print(prediction_label)

model_path = os.path.join(dirname, '../models/diabetes_model_nb.pkl')
joblib.dump(model, model_path)
print("Model saved as 'diabetes_model_nb.pkl'.")