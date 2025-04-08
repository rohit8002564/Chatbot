import os
import csv
import datetime
import streamlit as st
import json
import random
from gtts import gTTS
import speech_recognition as sr
import tempfile
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the JSON file
with open("nlpintents.json") as file:  # Use relative path
    data = json.load(file)

# Preprocess data
tags = []
patterns = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

x = vectorizer.fit_transform(patterns)
y = tags  # Directly using tag names
clf.fit(x, y)

# Chat function
def chat(user_input):
    input_vector = vectorizer.transform([user_input])
    prediction = clf.predict(input_vector)[0]
    for intent in data['intents']:
        if intent['tag'] == prediction:
            return random.choice(intent['responses'])

# Voice Output using gTTS
def speak(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")

# Listen function
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write("You said:", text)
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            st.write("Sorry, the speech service is down.")
            return None

# Streamlit App
def main():
    counter = 0
    st.title("Chatbot using NLP (Intents-Based)")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome!")

        # Create chat log file if not exists
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1

        input_type = st.radio("Choose input method:", ["Text", "Voice"])

        if input_type == "Text":
            user_input = st.text_input("You:", key=f"user_input_{counter}")

            if user_input:
                response = chat(user_input)
                st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_{counter}")
                speak(response)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([user_input, response, timestamp])

                if response.lower() in ['goodbye', 'bye']:
                    st.write("Thank you for chatting with me. Have a great day!")
                    st.stop()

        elif input_type == "Voice":
            if st.button("Speak"):
                user_input = listen()
                if user_input:
                    response = chat(user_input)
                    st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_voice_{counter}")
                    speak(response)
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([user_input, response, timestamp])

                    if response.lower() in ['goodbye', 'bye']:
                        st.write("Thank you for chatting with me. Have a great day!")
                        st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("------")
        except FileNotFoundError:
            st.write("No conversation history found.")

    elif choice == "About":
        st.subheader("Project Goal:")
        st.write("Develop an NLP chatbot that responds to user inputs using ML.")

        st.subheader("Project Overview:")
        st.write("""
        - **NLP + ML**: Trained with Logistic Regression and TF-IDF.
        - **Streamlit UI**: Clean, real-time interaction via text or voice.
        """)

        st.subheader("Dataset:")
        st.write("Contains patterns, tags, and responses for training the chatbot.")

if __name__ == '__main__':
    main()
