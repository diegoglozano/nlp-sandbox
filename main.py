import streamlit as st
import random

from utils.constants import SEED, NROWS, LIMITW2V
from utils.paths import DATA_PATH
from utils.helpers import read_data, read_word2vec, read_stop_words

from modules.introduction import introduction
from modules.logistic_regression import logistic_regression
from modules.deep_learning import deep_learning


# ----------------------------------------------------------------------------------------------------------------------

random.seed(SEED)

# ----------------------------------------------------------------------------------------------------------------------

st.balloons()

st.sidebar.title("NLP Sandbox")
st.sidebar.subheader("Choose your style :sunglasses:")
choice = st.sidebar.radio('',
                          ('Introduction', 'Logistic Regression', 'Deep Learning'))

df = read_data(DATA_PATH / 'train.csv', nrows=NROWS)
stop_words = read_stop_words(DATA_PATH / 'stop-words.txt')
word2vec = read_word2vec(DATA_PATH / 'word2vec.bin',
                         limit=LIMITW2V)

# ----------------------------------------------------------------------------------------------------------------------

if choice == 'Introduction':
    introduction(df, stop_words)

# ----------------------------------------------------------------------------------------------------------------------

if choice == 'Logistic Regression':
    logistic_regression(df, stop_words)

# ----------------------------------------------------------------------------------------------------------------------

if choice == 'Deep Learning':
    deep_learning(word2vec)
