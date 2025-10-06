from preprocess import preprocess_data
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Memuat dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target
data.head()
input_example = data[0:5]

# Contoh Penggunaan
X_train, X_test, y_train, y_test = preprocess_data(data, 'target', 'preprocessor_pipeline.joblib', 'data.csv')

y_train = np.array(y_train)
y_test = np.array(y_test)

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Create a new MLflow Experiment
mlflow.set_experiment("Latihan Model Statis")

with mlflow.start_run():
    mlflow.autolog()
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 505, 700],
        'max_depth': [5, 10, 15, 20, 25, 37, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize RandomForestClassifier
    rf = RandomForestClassifier()

    # Perform grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Log best parameters
    mlflow.log_params(best_params)

    # Train the best model on the training set
    best_model.fit(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=input_example
    )

    # Evaluate the model on the test set and log accuracy
    accuracy = best_model.score(X_test, y_test)