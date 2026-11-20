import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="DecisionTree"):
    dt = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
    acc1 = accuracy_score(y_test, dt.predict(X_test))
    mlflow.log_param("max_depth", 3)
    mlflow.log_metric("accuracy", acc1)

with mlflow.start_run(run_name="RandomForest"):
    rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    acc2 = accuracy_score(y_test, rf.predict(X_test))
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc2)

print("Decision Tree Accuracy =", acc1)
print("Random Forest Accuracy =", acc2)
print("Best Model :", "Decision Tree" if acc1 > acc2 else "Random Forest")
