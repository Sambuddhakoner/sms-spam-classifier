import pickle
import string
import nltk
nltk.download('punkt')
import streamlit as st
from nltk.data import load
from nltk.tokenize.destructive import NLTKWordTokenizer


# Implementing my_word_tokenize function


# Standard sentence tokenizer.
def sent_tokenize(text, language="english"):
    tokenizer = load(f"tokenizers/punkt/{language}.pickle")
    return tokenizer.tokenize(text)



# Standard word tokenizer.
_treebank_word_tokenizer = NLTKWordTokenizer()


def my_word_tokenize(text, language="english", preserve_line=False):
    sentences = [text] if preserve_line else sent_tokenize(text, language)
    return [
        token for sent in sentences for token in _treebank_word_tokenizer.tokenize(sent)
    ]


from nltk.stem.porter import PorterStemmer

stopwords = ['i',
             'me',
             'my',
             'myself',
             'we',
             'our',
             'ours',
             'ourselves',
             'you',
             "you're",
             "you've",
             "you'll",
             "you'd",
             'your',
             'yours',
             'yourself',
             'yourselves',
             'he',
             'him',
             'his',
             'himself',
             'she',
             "she's",
             'her',
             'hers',
             'herself',
             'it',
             "it's",
             'its',
             'itself',
             'they',
             'them',
             'their',
             'theirs',
             'themselves',
             'what',
             'which',
             'who',
             'whom',
             'this',
             'that',
             "that'll",
             'these',
             'those',
             'am',
             'is',
             'are',
             'was',
             'were',
             'be',
             'been',
             'being',
             'have',
             'has',
             'had',
             'having',
             'do',
             'does',
             'did',
             'doing',
             'a',
             'an',
             'the',
             'and',
             'but',
             'if',
             'or',
             'because',
             'as',
             'until',
             'while',
             'of',
             'at',
             'by',
             'for',
             'with',
             'about',
             'against',
             'between',
             'into',
             'through',
             'during',
             'before',
             'after',
             'above',
             'below',
             'to',
             'from',
             'up',
             'down',
             'in',
             'out',
             'on',
             'off',
             'over',
             'under',
             'again',
             'further',
             'then',
             'once',
             'here',
             'there',
             'when',
             'where',
             'why',
             'how',
             'all',
             'any',
             'both',
             'each',
             'few',
             'more',
             'most',
             'other',
             'some',
             'such',
             'no',
             'nor',
             'not',
             'only',
             'own',
             'same',
             'so',
             'than',
             'too',
             'very',
             's',
             't',
             'can',
             'will',
             'just',
             'don',
             "don't",
             'should',
             "should've",
             'now',
             'd',
             'll',
             'm',
             'o',
             're',
             've',
             'y',
             'ain',
             'aren',
             "aren't",
             'couldn',
             "couldn't",
             'didn',
             "didn't",
             'doesn',
             "doesn't",
             'hadn',
             "hadn't",
             'hasn',
             "hasn't",
             'haven',
             "haven't",
             'isn',
             "isn't",
             'ma',
             'mightn',
             "mightn't",
             'mustn',
             "mustn't",
             'needn',
             "needn't",
             'shan',
             "shan't",
             'shouldn',
             "shouldn't",
             'wasn',
             "wasn't",
             'weren',
             "weren't",
             'won',
             "won't",
             'wouldn',
             "wouldn't"]

punctuations = string.punctuation
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = my_word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y.copy()
    y.clear()

    for i in text:
        if i not in stopwords and i not in punctuations:
            y.append(i)
    text = y.copy()
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = cv.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
