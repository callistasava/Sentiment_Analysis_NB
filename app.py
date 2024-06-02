
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import time

model = pickle.load(open('sentiment.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_tfidf.pkl', 'rb'))

st.title('Review Apps Sentiment Analysis')

coms = st.text_input('Enter your commentary about the app')

submit = st.button('Predict')

if submit:
    start = time.time()
    # Transform the input text using the loaded TF-IDF vectorizer
    transformed_text = vectorizer.transform([coms]).toarray()
    st.write('Transformed text shape:', transformed_text.shape)  # Debugging statement
    # Reshape the transformed text to 2D array
    transformed_text = transformed_text.reshape(1, -1)
    st.write('Reshaped text shape:', transformed_text.shape)  # Debugging statement
    # Make prediction
    prediction = model.predict(transformed_text)
    end = time.time()
    st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
    
    print(prediction[0])
    if prediction[0] == 1:
        st.write("Sentimen review anda positif")
    else:
        st.write("Sentimen review anda negatif")
