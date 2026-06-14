import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

dagshub.init(
    repo_owner="annamaulina654",
    repo_name="MLSystem-Telco-Churn",
    mlflow=True
)

# Load dataset
df = pd.read_csv("telco_churn_processed.csv")

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Hyperparameter tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

with mlflow.start_run():

    # parameter
    mlflow.log_params(grid_search.best_params_)

    # metric
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # model
    mlflow.sklearn.log_model(
        best_model,
        "model"
    )

    # artifact 1
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    # artifact 2
    metric_info = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    with open("metric_info.json", "w") as f:
        json.dump(metric_info, f, indent=4)

    mlflow.log_artifact("metric_info.json")

    # artifact 3 - feature importance
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    })

    feature_importance.to_csv(
        "feature_importance.csv",
        index=False
    )

    mlflow.log_artifact(
        "feature_importance.csv"
    )

    # artifact 4 - best parameter
    with open(
        "best_params.json",
        "w"
    ) as f:
        json.dump(
            grid_search.best_params_,
            f,
            indent=4
        )

    mlflow.log_artifact(
        "best_params.json"
    )

print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)