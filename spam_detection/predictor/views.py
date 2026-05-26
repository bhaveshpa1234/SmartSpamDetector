import os
import joblib
import string
import nltk

from django.shortcuts import render

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Create stemmer object
ps = PorterStemmer()

# Store stopwords
stop_words = set(stopwords.words('english'))

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model
model = joblib.load(
    os.path.join(BASE_DIR, '../ml_model/spam_model.pkl')
)

# Load vectorizer
vectorizer = joblib.load(
    os.path.join(BASE_DIR, '../ml_model/vectorizer.pkl')
)


# Text preprocessing function
def transform_text(text):

    text = text.lower()

    text = word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Main page
def home(request):

    prediction = None

    if request.method == 'POST':

        message = request.POST.get('message')

        transformed_message = transform_text(message)

        vector_input = vectorizer.transform(
            [transformed_message]
        )

        result = model.predict(vector_input)[0]

        if result == 1:
            prediction = "Spam Message"
        else:
            prediction = "Ham Message"

    return render(
        request,
        'index.html',
        {'prediction': prediction}
    )