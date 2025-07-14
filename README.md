# Text_Emotion_detection

This project detects emotions (like joy, sadness, anger, etc.) from a given piece of text using Natural Language Processing (NLP) and a deep learning model (LSTM).

It was built using the dair-ai/emotion dataset and includes full preprocessing, model training, and a Gradio-based web interface.
It detects emotion in a class of 6 -> [sadness,joy,love,anger,fear,surprise]

Project Structure:

-> prepro.ipynb           # Text preprocessing

-> emo_lstm_model.ipynb   # Model building and training

-> gradio_app.py          # Gradio UI code

-> lstm_emo_model.h5      # Trained LSTM model

-> tokenizer.pkl          # Saved tokenizer

-> label_encoder.pkl      # Saved label encoder

─> README.md              # Project documentation

## How It Works:
Text Cleaning:
Lowercasing, punctuation removal, lemmatization.

Tokenization + Padding:
Keras Tokenizer + pad_sequences to convert text into numeric form.

Model:
A 2-layer LSTM with dropout and dense layers trained using Keras.

Inference:
A saved .h5 model is loaded and used in a Gradio web UI.

## Running the Gradio App:

To launch the web interface locally:

pip install gradio tensorflow nltk
python gradio_app.py

## Demo Screenshot

Here is how the Gradio interface looks:

![Emotion Detection Demo](demo.png)
