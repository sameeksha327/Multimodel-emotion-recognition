import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

model = load_model("fusion_model.h5")

speech = np.load("speech_features.npy")
text = np.load("text_features.npy")
labels = np.load("labels.npy")

pred = model.predict([speech, text])
y_pred = pred.argmax(axis=1)

accuracy = accuracy_score(labels, y_pred)
print("Fusion Accuracy:", accuracy)

