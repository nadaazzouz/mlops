import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ===== 1. Load and prepare the data =====
df = pd.read_csv("churn_data.csv")

X = pd.get_dummies(
    df[["age", "salaire", "abonnement", "anciennete", "consommation"]],
    drop_first=True
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    df["churn"],
    test_size=0.2,
    random_state=42
)


# ===== 2. Choose the MLflow experiment =====
mlflow.set_experiment("churn-prediction")


# ===== 3. Enable automatic logging =====
# MLflow automatically records model parameters, metrics, and the model itself.
mlflow.sklearn.autolog(serialization_format="cloudpickle")


# ===== 4. Train the model =====
with mlflow.start_run(run_name="trial-auto"):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)  # automatic logging is triggered by fit()

    # Calculate and display the accuracy on the test data
    print("Test score:", model.score(X_test, y_test))