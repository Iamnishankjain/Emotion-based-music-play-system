import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Function to extract features from audio files
def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Dictionary for emotion labels in RAVDESS
emotion_dict = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Load dataset and extract features
emotions = []
features = []
dataset_path = 'ravdess'  # RAVDESS dataset folder
for folder in os.listdir(dataset_path):
    for file in os.listdir(os.path.join(dataset_path, folder)):
        file_path = os.path.join(dataset_path, folder, file)
        # Check if the filename contains enough parts when split by '-'
        parts = file.split('-')
        if len(parts) < 3:
            print(f"Skipping file: {file} (unexpected format)")
            continue
        
        try:
            emotion = parts[2]  # E.g., '05' for 'angry'
            emotions.append(int(emotion) - 1)  # Convert to 0-based index for training
            features.append(extract_features(file_path))
        except ValueError:
            print(f"Skipping file: {file} (unable to parse emotion)")

X = np.array(features)
y = np.array(emotions)

# Check if features and labels were extracted successfully
if len(X) == 0 or len(y) == 0:
    raise ValueError("No valid audio data found. Check the dataset and filenames.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build the model
voice_model = Sequential([
    Dense(256, input_shape=(40,), activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(8, activation='softmax')  # 8 emotion classes in RAVDESS
])

voice_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
voice_model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save the model
voice_model.save('voiceemotionmodel.h5')
