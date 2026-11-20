import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame({
    'Student_ID': [1, 2, 3, 4, 5, 6],
    'Hours': [2, 4, 6, 8, 3, 7],
    'Attendance': [50, 65, 80, 95, 55, 90],
    'Pass': [0, 0, 1, 1, 0, 1]
})

feature_store = {}
for i, row in df.iterrows():
    feature_store[row['Student_ID']] = [row['Hours'], row['Attendance']]

print("Feature Store:")
print(feature_store)
print("\nFeatures of Student 1:")
print(feature_store[1])

X = df[['Hours', 'Attendance']]
y = df['Pass']

model1 = DecisionTreeClassifier(random_state=42).fit(X, y)
model2 = LogisticRegression().fit(X, y)

print("\nDecision Tree Model Trained")
print("Logistic Regression Model Trained")

student = pd.DataFrame([feature_store[1]], columns=['Hours', 'Attendance'])

print("\nPrediction using Decision Tree:", model1.predict(student))
print("Prediction using Logistic Regression:", model2.predict(student))
