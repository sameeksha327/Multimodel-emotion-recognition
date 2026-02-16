import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

model = load_model("text_model.h5")

X = np.load("text_features.npy")
y = np.load("text_labels.npy")

pred = model.predict(X)
y_pred = pred.argmax(axis=1)

accuracy = accuracy_score(y, y_pred)
print("Text Accuracy:", accuracy)

