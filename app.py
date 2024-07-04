import streamlit as st
import pickle
import string
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
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

# Load your vectorizer and model
tfidf = pickle.load(open('vector.pkl', 'rb'))
model = pickle.load(open('model2.pkl', 'rb'))

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Input text area for the message
input_sms = st.text_area("Enter the message")

# Predict button
if st.button('Predict'):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict
    result = model.predict(vector_input)[0]
    
    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
