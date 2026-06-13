import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load Dataset
df = pd.read_csv("telco_churn_processed.csv")

# Split Feature dan Target
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# MLflow Autolog
mlflow.autolog()

with mlflow.start_run():

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(f"Accuracy: {accuracy:.4f}")