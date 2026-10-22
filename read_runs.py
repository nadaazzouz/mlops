import mlflow  # we reuse the MLflow module to READ the results

# Search for all runs of the experiment
# Sort by F1 score in descending order (best first)
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# Columns to display: run name, two parameters, and two metrics
columns = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1"
]

# Display the results without row numbers
print(runs[columns].to_string(index=False))

# The first row is the best run
best = runs.iloc[0]

print()

# Display the best run's name and the first 8 characters of its ID
print(
    "Best run :",
    best["tags.mlflow.runName"],
    "| id :",
    best["run_id"][:8],
    "..."
)