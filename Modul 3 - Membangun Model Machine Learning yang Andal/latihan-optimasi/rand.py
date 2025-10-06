from sklearn.model_selection import ParameterSampler
from scipy.stats import randint
from base import train_and_evaluate_model
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data menjadi training dan testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tentukan distribusi hyperparameter
param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 11)
}

# Tentukan jumlah iterasi
n_iter_search = 10

# Mulai eksperimen MLflow
with mlflow.start_run(run_name="Random Search Experiment"):
    # Iterasi melalui kombinasi hyperparameter yang dipilih secara acak
    for params in ParameterSampler(param_distributions, n_iter=n_iter_search):
        # Latih dan evaluasi model untuk setiap kombinasi
        train_and_evaluate_model(params, X_train, y_train, X_test, y_test)
