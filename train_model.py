import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('fer2013.csv')

# Preprocess the dataset
def preprocess_data(data):
    X = []
    y = []
    for i in range(len(data)):
        pixels = data['pixels'][i].split()
        X.append(np.array(pixels, dtype='float32'))
        y.append(data['emotion'][i])
    X = np.array(X)
    X = X.reshape(-1, 48, 48, 1) / 255.0  # Normalize and reshape to (48,48,1)
    y = np.array(y)
    return X, y

X, y = preprocess_data(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # 7 emotion classes
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save the trained model
model.save('emotion_model.h5')
