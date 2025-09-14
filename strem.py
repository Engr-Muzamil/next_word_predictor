import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('model_word_lstm.h5', compile=False)
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

total_words = len(tokenizer.word_index) + 1
max_seq_len = model.input_shape[1] or 10  # fallback if None

st.title("Next Word Prediction")

# Get user text
user_text = st.text_input("Enter your text:")

if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([user_text])[0]
    seq = pad_sequences([seq], maxlen=max_seq_len-1, padding='pre')
    pred = np.argmax(model.predict(seq), axis=-1)[0]

    next_word = tokenizer.index_word.get(pred, "unknown")
    st.write(f"**Next Word Prediction:** {next_word}")

 