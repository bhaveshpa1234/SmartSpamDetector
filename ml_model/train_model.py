import pandas as pd
import nltk
import string
import joblib

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Create stemmer object
ps = PorterStemmer()

# Store stopwords once (Optimization)
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

# Show first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Check null values
print("\nNull Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates(keep='first')

print("\nShape After Removing Duplicates:")
print(df.shape)

# Convert labels
# ham = 0
# spam = 1
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

print("\nConverted Labels:")
print(df.head())


# Text preprocessing function
def transform_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Tokenization
    text = word_tokenize(text)

    y = []

    # Remove special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Apply preprocessing
df['transformed_message'] = df['message'].apply(transform_text)

print("\nTransformed Dataset:")
print(df.head())


# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_message']).toarray()

y = df['label'].values


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)


# Create Naive Bayes model
model = MultinomialNB(alpha=0.1)


# Train model
model.fit(X_train, y_train)


# Predict test data
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)


# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Test custom message
sample = "Congratulations! You won a free ticket"


# Preprocess sample message
transformed_sample = transform_text(sample)


# Convert text into vector
vector_input = tfidf.transform([transformed_sample])


# Prediction
result = model.predict(vector_input)[0]


print("\nCustom Message Prediction:")

if result == 1:
    print("Spam Message")
else:
    print("Ham Message")

# Save model
joblib.dump(model, 'spam_model.pkl')

# Save TF-IDF vectorizer
joblib.dump(tfidf, 'vectorizer.pkl')

print("\nModel and Vectorizer Saved Successfully")