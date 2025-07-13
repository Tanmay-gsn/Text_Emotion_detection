import gradio as gr
import numpy as np
import re
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load model and files
model = load_model("/kaggle/input/emo-lstm/lstm_emo_model.h5")

with open("/kaggle/input/emo-lstm/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("/kaggle/input/emo-lstm/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Preprocessing
lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized)

# ✅ Use correct label order for dair-ai/emotion
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def predict_emotion(text):
    try:
        cleaned_text = clean_and_lemmatize(text)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=100)
        pred_probs = model.predict(padded)
        pred_index = np.argmax(pred_probs, axis=1)[0]
        pred_label = emotion_labels[pred_index]
        return str(pred_label)
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=2, placeholder="Type a sentence..."),
    outputs=gr.Textbox(label="Predicted Emotion"),
    title="Emotion Detection with LSTM",
    description="Enter a sentence to detect emotions like joy, anger, sadness, etc.",
)

if __name__ == "__main__":
    iface.launch()