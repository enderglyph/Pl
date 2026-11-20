import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame({
    'Hours': [2, 4, 6, 8, 3, 7],
    'Marks': [40, 45, 65, 80, 38, 78],
    'Pass': [0, 0, 1, 1, 0, 1]
})
print("Original Data:")
print(df)

df['Total'] = df['Hours'] + df['Marks']

print("\nMissing Values:")
print(df.isnull().sum())

X = df[['Hours', 'Marks', 'Total']]
y = df['Pass']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
]).fit(X, y)

prediction = pipeline.predict(pd.DataFrame({'Hours': [5], 'Marks': [60], 'Total': [65]}))

print("\nPrediction:", prediction[0])
print("Pipeline Executed Successfully")
