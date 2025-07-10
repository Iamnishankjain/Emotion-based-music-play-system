from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import speech_recognition as sr
from googleapiclient.discovery import build
import webbrowser
import threading

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models
emotion_model = load_model('emotion_model.h5')
voice_model = load_model('voiceemotionmodel.h5')

# Emotion labels (for the FER2013 dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# YouTube API Key
YOUTUBE_API_KEY = 'YOUTUBE_API_KEY'     //replace it

# Set up YouTube API client
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Global variables to track status
is_face_detection_active = False
is_voice_detection_active = False
is_video_playing = False
current_video_url = ""

# Start video capture for face detection
cap = cv2.VideoCapture(0)

# Query to search YouTube based on emotion
def search_youtube_for_songs(emotion):
    global is_video_playing, current_video_url
    search_queries = {
        'Happy': 'Bollywood happy song',
        'Sad': 'Arijit Singh sad song',
        'Angry': 'Bollywood angry song',
        'Fear': 'Bollywood fear song',
        'Surprise': 'Bollywood surprise song',
        'Disgust': 'Bollywood disgust song',
        'Neutral': 'Bollywood relaxing song'
    }

    query = search_queries.get(emotion, 'Bollywood relaxing song')

    request = youtube.search().list(
        part='snippet',
        q=query,
        type='video',
        maxResults=1
    )
    response = request.execute()

    if 'items' in response and len(response['items']) > 0:
        video_id = response['items'][0]['id']['videoId']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        print(f"Playing {emotion} music: {video_url}")

        if not is_video_playing:
            webbrowser.open_new(video_url)
            is_video_playing = True
            current_video_url = video_url
        else:
            print("A video is already playing, skipping this one.")
    else:
        print("No video found for the emotion.")

# Stop face detection after a video is played
def stop_face_detection():
    global is_face_detection_active
    print("Stopping face detection and closing the video.")
    is_face_detection_active = False

# Detect face and emotion using the pre-trained model
def detect_face_emotion(frame):
    global is_face_detection_active
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

        # Stop face detection after detecting an emotion
        stop_face_detection()

    return frame

# Generate video frames for Flask
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect face emotion if detection is active
        frame = detect_face_emotion(frame)

        # Encode the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Yield the frame as a byte stream for browser display
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# Voice emotion detection function
def detect_voice_emotion():
    global is_voice_detection_active
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

            # Stop voice detection after detecting emotion
            is_voice_detection_active = False
        except Exception as e:
            print(f"Error recognizing voice: {str(e)}")

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_face_detection')
def start_face_detection():
    global is_face_detection_active
    is_face_detection_active = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_face_detection')
def stop_face_detection_route():
    global is_face_detection_active
    is_face_detection_active = False
    return "Stopped Face Detection"

@app.route('/start_voice_detection')
def start_voice_detection():
    global is_voice_detection_active
    is_voice_detection_active = True
    detect_voice_emotion()
    return "Voice Detection Started"

if __name__ == "__main__":
    app.run(debug=True)
