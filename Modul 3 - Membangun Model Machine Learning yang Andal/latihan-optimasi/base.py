import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fungsi untuk melatih dan mengevaluasi model
def train_and_evaluate_model(params, X_train, y_train, X_test, y_test):
    with mlflow.start_run(nested=True):  # nested=True untuk hyperparameter tuning
        # Inisialisasi model dengan hyperparameter yang diberikan
        model = RandomForestClassifier(**params, random_state=42)

        # Latih model
        model.fit(X_train, y_train)

        # Prediksi pada data test
        y_pred = model.predict(X_test)

        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameter dan metrik ke MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Log model ke MLflow (opsional)
        mlflow.sklearn.log_model(model, "model")

        return accuracy