import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
import streamlit as st 
import pickle

model=load_model('next_word.h5')

#load tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)


def predict_next_word(model, tokenizer, text, max_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_len:
        token_list = token_list[-(max_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


#streamlit app
st.title("Next Word Prediction using RNN")
st.write("Enter a partial sentence to predict the next word:")
input_text = st.text_area("Input Text","To be or not to be ")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Predicted Next Word: **{next_word}**')


