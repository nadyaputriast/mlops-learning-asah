from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
import mlflow
from base import train_and_evaluate_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data menjadi training dan testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tentukan ruang pencarian hyperparameter
space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 1)),
    'max_depth': hp.choice('max_depth', [None, scope.int(hp.quniform('max_depth_int', 10, 20, 1))]),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1))
}

# Fungsi objektif untuk Hyperopt
def objective(params):
    with mlflow.start_run(nested=True):
        accuracy = train_and_evaluate_model(params, X_train, y_train, X_test, y_test)
        # Hyperopt meminimalkan fungsi objektif, jadi kita gunakan -accuracy
        return {'loss': -accuracy, 'status': STATUS_OK}

# Mulai eksperimen MLflow
with mlflow.start_run(run_name="Bayesian Optimization Experiment"):
    # Jalankan optimisasi Bayesian
    # use patience (early stopping) to avoid overfitting
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=10,  # Jumlah evaluasi
                trials=trials)

    # Log hyperparameter terbaik ke MLflow
    best_params = {k: v[0] if isinstance(v, list) else v for k, v in trials.best_trial['misc']['vals'].items()}
    mlflow.log_params(best_params)