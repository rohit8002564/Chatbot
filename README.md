## Internship Experience

**Virtual Internship - Artificial Intelligence & Data Analytics (Green Skills)**  
Edunet Foundation | AICTE | Shell India — *Skills4Future Program*  
📅 Dec 2024 – Jan 2025 (4 weeks)

- Successfully completed a virtual internship focused on AI, Data Analytics, and Green Skills.  
- Developed a chatbot using Natural Language Processing (NLP) under the guidance of an industry expert.  
- Gained hands-on experience in real-world project implementation and AI tools.

## Certificate
[![Certificate](https://img.shields.io/badge/Certificate-View-green)](https://drive.google.com/drive/folders/1i84lrguZVxif4yUWe_A1xAlXD5m30RIp?usp=sharing)



## 🤖 NLP Chatbot (Intent-Based)
<img width="1915" height="919" alt="Screenshot 2025-10-08 102028" src="https://github.com/user-attachments/assets/00ff98da-d7d8-47cc-b3da-92b6777b9813" />
This is a simple **NLP-based chatbot** that uses **Machine Learning** to understand user intents and respond accordingly. Built with **Streamlit** for a web interface, it supports **text input** and responds with **voice output (gTTS)**.

## 🔗 Live Demo

👉 [Click here to try the chatbot live!](https://chatbot-am6wggwizydzsdzkag8kyz.streamlit.app/)



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
