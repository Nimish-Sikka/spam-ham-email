import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import time

ps = PorterStemmer()

def transforming_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Email/SMS Spam Classifier", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F0EAD6;
    }
    /* Style for input box */
    div.stTextInput>div>div>div>input {
        background-color: #BFE3F7 !important; /* Pastel blue */
        color: #4A2E0F !important; /* Dark brown */
    }
    /* Style for predict button */
    div.stButton>button {
        background-color: #BFE3F7 !important; /* Pastel blue */
        color: #4A2E0F !important; /* Dark brown */
    }
    /* Style for heading text */
    h1, h2, h3, h4, h5, h6 {
        color: #4A2E0F !important; /* Dark brown */
    }
    /* Style for other texts */
    body, p, span, label {
        color: #4A2E0F !important; /* Dark brown */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Email/SMS Spam Classifier")
st.markdown("---")

input_sms = st.text_area("Enter the message", height=200)

if st.button('Predict'):
    # Preprocess the input
    transformed_sms = transforming_text(input_sms)
    # Vectorize the input
    vector_input = tfidf.transform([transformed_sms])
    # Make prediction
    result = model.predict(vector_input)[0]
    if result == 1:
        st.error("**❌!!SPAM!!❌**")
    else:
        st.success("**💚NOT SPAM💚**")
