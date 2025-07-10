import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import speech_recognition as sr
from googleapiclient.discovery import build
import webbrowser
import threading

# Load the pre-trained models
emotion_model = load_model('emotion_model.h5')
voice_model = load_model('voiceemotionmodel.h5')

# Load labels for emotions (for the FER2013 dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# YouTube API Key
YOUTUBE_API_KEY = 'YOUTUBE_API_KEY'     //replace it

# Set up YouTube API client
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Global variables to track if a video is playing and if face detection is active
is_video_playing = False
current_video_url = ""
is_face_detection_active = True  # Start with face detection active

# Function to search YouTube for songs based on emotion
def search_youtube_for_songs(emotion):
    global is_video_playing, current_video_url, is_face_detection_active

    # If a video is already playing, stop it
    if is_video_playing:
        print(f"Stopping the currently playing video: {current_video_url}")
        # Close the browser tab or window to stop the current video
        webbrowser.open('about:blank')  # Close previous tab (this is a workaround)
        is_video_playing = False
        current_video_url = ""

    # Search queries for different emotions (Bollywood and Arijit Singh only)
    search_queries = {
        'Happy': 'Bollywood happy song',
        'Sad': 'Arijit Singh sad song',
        'Angry': 'Bollywood angry song',
        'Fear': 'Bollywood fear song',
        'Surprise': 'Bollywood surprise song',
        'Disgust': 'Bollywood disgust song',
        'Neutral': 'Bollywood relaxing song'
    }

    # Get the search query based on the emotion
    query = search_queries.get(emotion, 'Bollywood relaxing song')

    # Search YouTube for the top video related to the emotion
    request = youtube.search().list(
        part='snippet',
        q=query,
        type='video',
        maxResults=1
    )
    response = request.execute()

    # If there are items in the response, extract the video ID and URL
    if 'items' in response and len(response['items']) > 0:
        video_id = response['items'][0]['id']['videoId']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        print(f"Playing {emotion} music: {video_url}")

        # Open the new video URL in the default browser
        webbrowser.open_new(video_url)
        is_video_playing = True
        current_video_url = video_url

        # Set a timer to stop face detection and close the video after playing
        threading.Timer(10, stop_face_detection).start()  # Stop after 10 seconds

    else:
        print("No video found for the emotion.")

# Function to stop face detection and close the video after playing the song
def stop_face_detection():
    global is_face_detection_active
    print("Stopping face detection and closing the video.")
    is_face_detection_active = False

# Function to detect face emotion
def detect_face_emotion(frame):
    global is_face_detection_active

    # Only detect faces if face detection is active
    if not is_face_detection_active:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Predict emotion
        emotion_prediction = emotion_model.predict(face)
        emotion_index = np.argmax(emotion_prediction)
        emotion = emotion_labels[emotion_index]

        # Draw rectangle and emotion label
        color = (0, 255, 0) if emotion == 'Happy' else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Play song based on detected emotion
        search_youtube_for_songs(emotion)

        # Stop face detection after the first song is played
        stop_face_detection()

    return frame

# Function to detect voice emotion
def detect_voice_emotion():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening for voice...")
        audio_data = recognizer.listen(source)
        print("Recognizing voice...")

        try:
            audio_text = recognizer.recognize_google(audio_data)
            print(f"Recognized text: {audio_text}")
            audio_features = np.array([audio_text])

            # Predict voice emotion
            voice_prediction = voice_model.predict(audio_features)
            voice_emotion = np.argmax(voice_prediction)
            voice_emotion_label = emotion_labels[voice_emotion]
            print(f"Voice Emotion Detected: {voice_emotion_label}")

            # Play song based on voice emotion
            search_youtube_for_songs(voice_emotion_label)

            # Stop face detection after the first song is played
            stop_face_detection()

        except Exception as e:
            print(f"Error recognizing voice: {str(e)}")

# Main function to run the emotion-sensitive workspace
def run_emotion_workspace():
    # Start the webcam for face detection
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Detect face emotion if face detection is active
        frame = detect_face_emotion(frame)

        # Show the live video feed
        cv2.imshow('Emotion Sensitive Workspace', frame)

        # If 'q' is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Detect voice emotion
    # detect_voice_emotion()

    # Start the emotion-sensitive workspace
    run_emotion_workspace()
