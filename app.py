from flask import Flask, render_template, request
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data safely
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()


# Text preprocessing function
def transform_text(text):

    # Convert to lowercase
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text, preserve_line=True)

    y = []

    # Remove special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    stop_words = stopwords.words('english')

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Home Route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    # Get input message
    input_sms = request.form['message']

    # Transform text
    transformed_sms = transform_text(input_sms)

    # Vectorize text
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Display result
    if result == 1:
        prediction = "Spam 🚨"
    else:
        prediction = "Not Spam ✅"

    return render_template(
        'index.html',
        prediction=prediction
    )


# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)