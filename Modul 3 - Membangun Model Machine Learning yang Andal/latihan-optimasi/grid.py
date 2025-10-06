from sklearn.model_selection import ParameterGrid
from base import train_and_evaluate_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mlflow

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data menjadi training dan testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Tentukan grid hyperparameter
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Mulai eksperimen MLflow
with mlflow.start_run(run_name="Grid Search Experiment"):
    # Iterasi melalui semua kombinasi hyperparameter dalam grid
    for params in ParameterGrid(param_grid):
        # Latih dan evaluasi model untuk setiap kombinasi
        train_and_evaluate_model(params, X_train, y_train, X_test, y_test)
