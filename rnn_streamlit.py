from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence 
import numpy as np
import tensorflow as tf 
import streamlit as st
from tensorflow.keras.models import load_model
st.title("IMDB Reviews And Sentiment Analysys!")
word_index=imdb.get_word_index()
reverse_word_index={value:key for key ,value in word_index.items()}
model = load_model("imdb_review_che2.h5")

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#Fn preprocess user input
import re

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    words = text.split()

    encoded_review = []

    for word in words:
        if word in word_index:
            encoded_review.append(word_index[word] + 3)
        else:
            encoded_review.append(2)   # Unknown token

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review
def prediction_sentiment(review):
    preprocess_input=preprocess_text(review)
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] >= 0.5 else 'Negative'
    return sentiment,prediction[0][0]

st.write("Enter a movie review to classify it as positive or negative.")
input_data=st.text_area("Write your Review Here.....")

if st.button('Classify'):
    preprocess_input=preprocess_text(input_data)
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Predicted score: {prediction[0][0]}')
else:
    st.write("Enter your review first")