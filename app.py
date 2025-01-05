import os
import csv
import datetime
import streamlit as st
import json
import random
import pyttsx3
import speech_recognition as sr
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Initialize the voice engine
engine = pyttsx3.init()

# Load the json file
with open("C:/Users/Rohit kumar/Downloads/nlpintents.json") as file:
    data = json.load(file)

# Preprocess data
tags = []
patterns = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Debugging: Print patterns and tags to check the data
print("Patterns:", patterns[:5])  # Print first 5 patterns
print("Tags:", tags[:5])  # Print first 5 tags

# Initialize the TfidfVectorizer and Logistic Regression classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Transform patterns using TfidfVectorizer
x = vectorizer.fit_transform(patterns)

# Debugging: Check the shape of the transformed data
print("Shape of feature matrix:", x.shape)

# Encode the tags (if needed, use LabelEncoder for non-numerical tags)
y = tags

# Debugging: Check the length of y
print("Length of tags:", len(y))

# Train a Logistic Regression classifier
clf.fit(x, y)

# Define chat function
def chat(user_input):
    input_vector = vectorizer.transform([user_input])
    prediction = clf.predict(input_vector)[0]  # Use clf instead of model
    for intent in data['intents']:
        if intent['tag'] == prediction:
            return random.choice(intent['responses'])

counter = 0

def speak(text):
    """Function to make the chatbot speak in a separate thread."""
    def run_speech():
        engine.say(text)  # Uncommented the line for speech
        engine.runAndWait()

    # Run the speech in a separate thread
    threading.Thread(target=run_speech).start()

def listen():
    """Function to listen for user speech."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said: ", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Sorry, the speech service is down.")
            return None

def main():
    global counter
    st.title("Intents of Chatbot using NLP")
    
    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Home Menu
    if choice == "Home":
        st.write("Welcome")
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])
                
        counter += 1
        
        # Option to toggle between text or voice
        input_type = st.radio("Choose input method:", ["Text", "Voice"])
        
        if input_type == "Text":
            # Text input (chat)
            user_input = st.text_input("You:", key=f"user_input_{counter}")
            
            if user_input:
                # Process text input
                user_input_str = str(user_input)
                response = chat(user_input_str)
                st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_{counter}")
                speak(response)  # Convert the response to speech
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([user_input_str, response, timestamp])
                
                if response.lower() in ['goodbye', 'bye']:
                    st.write("Thank you for chatting with me. Have a great day!")
                    st.stop()

        elif input_type == "Voice":
            # Voice input
            voice_input_button = st.button("Speak")
            if voice_input_button:
                user_input = listen()
                if user_input:
                    st.text(f"You (Voice): {user_input}")
                    response = chat(user_input)
                    st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_voice_{counter}")
                    speak(response)  # Convert the response to speech
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow([user_input, response, timestamp])
                    
                    if response.lower() in ['goodbye', 'bye']:
                        st.write("Thank you for chatting with me. Have a great day!")
                        st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("------")
                
    elif choice == "About":
        st.subheader("Project Goal:") 
        st.write("The aim of this project is to develop a chatbot capable of understanding and responding to user inputs through Natural Language Processing (NLP) techniques, combined with machine learning algorithms for effective interaction.")
        st.subheader("Project Overview:")
        st.write("""
        This project is structured into two main components:
        
        1. NLP & Machine Learning: The chatbot is trained using NLP techniques and the Logistic Regression algorithm on labeled datasets to improve its ability to process and respond to user queries accurately.
        2. Streamlit Interface: Streamlit is utilized to create a user-friendly interface for seamless interaction with the chatbot, enabling users to easily input questions and receive responses in real time.
        """)
        
        st.subheader("Dataset:")
        st.write("The dataset contains intents, patterns, and responses used to train the chatbot, enabling it to accurately understand user queries and provide relevant responses.")

if __name__ == '__main__':
    main()
