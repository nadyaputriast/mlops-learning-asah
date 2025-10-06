import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import mlflow
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
# Load the mnist dataset.
train_ds, test_ds = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
)

def preprocess_fn(data):
    image = tf.cast(data["image"], tf.float32) / 255
    label = data["label"]
    return (image, label)


train_ds = train_ds.map(preprocess_fn).batch(128).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_fn).batch(128).prefetch(tf.data.AUTOTUNE)

input_shape = (28, 28, 1)
num_classes = 10

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(0.001),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("/mlflow-tf-keras-mnist")
with mlflow.start_run():
    mlflow.tensorflow.autolog()

    model.fit(x=train_ds, epochs=20)

    score = model.evaluate(test_ds)

    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]: .2f}")
    mlflow.log_metric("test_loss", score[0])
    mlflow.log_metric("test_accuracy", score[1])

    # Compute and log confusion matrix
    all_labels = []
    all_preds = []

    for images, labels in test_ds:
        preds = model.predict(images)
        all_labels.extend(labels.numpy())
        all_preds.extend(np.argmax(preds, axis=1))

    conf_matrix = confusion_matrix(all_labels, all_preds)
    mlflow.log_text(str(conf_matrix), "confusion_matrix.txt")

    # Log classification report
    class_report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)])
    mlflow.log_text(class_report, "classification_report.txt")