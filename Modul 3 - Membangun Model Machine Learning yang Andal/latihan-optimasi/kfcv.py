import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data menjadi training dan testing sets
# (Stratified) K-fold memerlukan semua data, train test split hanya untuk
# perbandingan model akhir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model
model = RandomForestClassifier(random_state=42)

# Tentukan jumlah fold
n_splits = 5

# Inisialisasi KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Mulai eksperimen MLflow
with mlflow.start_run(run_name="K-Fold Cross-Validation Experiment"):
    # Log parameter KFold
    mlflow.log_param("n_splits", n_splits)
    mlflow.log_param("shuffle", True)

    # Lakukan K-Fold Cross-Validation
    scores = []
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        with mlflow.start_run(nested=True, run_name=f"Fold-{fold + 1}"):
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            # Latih model
            model.fit(X_train_fold, y_train_fold)

            # Prediksi pada data validasi
            y_pred_fold = model.predict(X_val_fold)

            # Evaluasi model
            accuracy = accuracy_score(y_val_fold, y_pred_fold)
            scores.append(accuracy)

            # Log metrik untuk fold ini
            mlflow.log_metric("accuracy", accuracy)
            print(f"Fold-{fold + 1} accuracy: {accuracy}")

            # Log model untuk fold ini (opsional)
            mlflow.sklearn.log_model(model, f"model_fold_{fold + 1}")

    # Hitung rata-rata skor dari semua fold
    avg_accuracy = np.mean(scores)

    # Log rata-rata skor ke MLflow
    mlflow.log_metric("average_accuracy", avg_accuracy)
    print(f"\nAverage k-Fold accuracy: {avg_accuracy}")

    # Evaluasi model pada test set (di luar cross-validation)
    # Latih model dengan data latih yang dipisah diawal
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    mlflow.sklearn.log_model(model, "model final")

    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"Test accuracy: {test_accuracy}")