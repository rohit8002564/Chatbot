## 🤖 NLP Chatbot (Intent-Based)

This is a simple **NLP-based chatbot** that uses **Machine Learning** to understand user intents and respond accordingly. Built with **Streamlit** for a web interface, it supports **text input** and responds with **voice output (gTTS)**.

🔗 Live Demo
<p align="center">
  <a href="(https://chatbot-am6wggwizydzsdzkag8kyz.streamlit.app/)" target="_blank">
    <img src="https://img.shields.io/badge/Streamlit-Live_App-blue?style=for-the-badge&logo=streamlit" alt="Streamlit App">
  </a>
</p>


---

### 🚀 Features

- **Intent Recognition** using `TF-IDF` + `Logistic Regression`
- **Text Input** via a chat interface
- **Voice Response** using `gTTS` (Google Text-to-Speech)
- **Streamlit Web UI**
- **Conversation History** saved in a `.csv` file
- **Easy to Customize** – update `nlpintents.json` for new intents

---

### 🧠 How it works

1. Loads training data from `nlpintents.json`
2. Vectorizes the patterns using `TF-IDF`
3. Trains a `Logistic Regression` model to classify input into an intent
4. Responds with a random response from the matching intent
5. Speaks the response using `gTTS`

---

### 📂 Project Structure

```
📁 root/
│
├── .devcontainer/
│   └── devcontainer.json        # Dev container config (optional)
│
├── Chatbot.ipynb                # Jupyter Notebook for model development
├── README.md                    # Project documentation
├── app.py                       # Main Streamlit chatbot app
├── nlpintents.json              # Intents and responses
├── requirements.txt             # Python dependencies


---

### ✅ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```txt
streamlit
scikit-learn
gtts
speechrecognition
```

---

### ▶️ Run the App

```bash
streamlit run app.py
```

---

### 📘 Example JSON Format (`nlpintents.json`)

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey there"],
      "responses": ["Hello!", "Hi there!", "Hey! How can I help you?"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "Goodbye"],
      "responses": ["Goodbye!", "See you later!", "Take care!"]
    }
  ]
}
```

---

### 🛠️ Future Enhancements

- Add voice input using `SpeechRecognition`
- Improve intent classification with Deep Learning (RNN / BERT)
- Add contextual memory for conversations
