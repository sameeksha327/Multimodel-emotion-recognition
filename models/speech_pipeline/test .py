import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load trained model
model = load_model("speech_model.h5")

# Load test features and labels
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Predict
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# If y_test is one-hot encoded
if len(y_test.shape) > 1:
    y_test = np.argmax(y_test, axis=1)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Speech Model Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

