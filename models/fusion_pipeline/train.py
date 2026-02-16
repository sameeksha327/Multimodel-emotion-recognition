import numpy as np
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# Load features
speech = np.load("speech_features.npy")
text = np.load("text_features.npy")
labels = np.load("labels.npy")

y = to_categorical(labels)

# Inputs
speech_input = Input(shape=(speech.shape[1],))
text_input = Input(shape=(text.shape[1],))

# Combine
combined = Concatenate()([speech_input, text_input])

x = Dense(128, activation='relu')(combined)
x = Dense(64, activation='relu')(x)
output = Dense(y.shape[1], activation='softmax')(x)

model = Model(inputs=[speech_input, text_input], outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit([speech, text], y, epochs=20, batch_size=32)

model.save("fusion_model.h5")

print("Fusion model trained")

