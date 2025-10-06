from preprocess import preprocess_data
from sklearn.datasets import load_iris
import pandas as pd
import mlflow
from sklearn.linear_model import SGDClassifier
from joblib import dump


# Memuat dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target


# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")


# Create a new MLflow Experiment
mlflow.set_experiment("Online Training Iris")


# Model Online Learning: SGDClassifier
model = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, max_iter=10000)


# Kelas target (diperlukan untuk partial_fit)
classes = data['target'].unique()


with mlflow.start_run():
    mlflow.autolog()  # Melakukan logging otomatis


    # Preprocessing data untuk batch
    X_batch = data.drop(columns=['target'])
    y_batch = data['target']


    # Initial fit untuk batch pertama
    model.partial_fit(X_batch, y_batch, classes=classes)


    # Log metrik setelah setiap batch
    accuracy = model.score(X_batch, y_batch)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")
    # Simpan model ke file lokal
    dump(model, "online_model.joblib")

    # Log file model sebagai artefak ke MLflow
    mlflow.log_artifact("online_model.joblib", artifact_path="model_artifacts")

    # Log model setelah selesai online training
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="online_model",
        input_example=X_batch.iloc[:5]
    )
