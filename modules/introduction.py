import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer

from typing import Tuple

from utils.helpers import space, get_sentences_len, get_counts, plot_counter, plot_labels
from utils.constants import SEED


def introduction(df: pd.DataFrame,
                 stop_words: Tuple[str]):

    st.title('NLP Sandbox')
    st.write(f"""
            We're going to cover the fundamentals of Natural Language Processing. To achieve this, we'll use some
            data from Kaggle. It consists on plots from {df.shape[0]} films and their associated sentiment 
            (positive :thumbsup: or negative :thumbsdown:).
    
            First of all, here you have an introduction and a little analysis to the data that we are going to use.
    
            :point_left: Choose on the sidebar, what do you want to try?
        """)

    # ----------------------------------------------------------------------------------------------------------------------

    st.header('1. Read data')

    st.dataframe(df.head())
    st.text(f'DataFrame shape: {df.shape}')

    # ----------------------------------------------------------------------------------------------------------------------

    st.header('2. Analysis')

    st.markdown("""
        #### Let's see some texts
        """)
    space()

    sampled_text = df['text'].sample(3, random_state=SEED).values

    st.text(sampled_text[0])
    st.text(sampled_text[1])
    st.text(sampled_text[2])
    space()

    st.write('It seems there are some HTML symbols, like \'<br />\'. Later we\'ll take care of this.')

    st.markdown("""
        #### How many words has the texts?
        """)
    space()

    splitted_text = get_sentences_len(df.text)

    st.write("Max: ", splitted_text.max())
    st.write("Min: ", splitted_text.min())
    st.write("Average: ", splitted_text.mean().astype('int'))
    space()
    st.markdown(f"""
        {splitted_text.max()}? {splitted_text.min()}? Outliers? :fearful:  
        Let's check them:
        """)

    st.text(df.loc[splitted_text == splitted_text.max(), 'text'].values[0])
    st.text(df.loc[splitted_text == splitted_text.min(), 'text'].values[0])

    space()
    st.write('They seem to be real plots made by people with {too much|no} time.')

    st.markdown("""
        #### Which are the most common {words|pair of words}?
        """)

    st.markdown('For this purpose, we have deleted typical words, we\'ll talk later about that')
    counter = CountVectorizer(stop_words=stop_words,
                              ngram_range=(1, 2))
    counts_df, counter_feature_names = get_counts(counter, df.text)

    st.plotly_chart(plot_counter(counts_df))

    st.markdown('Wow! We have to take care of \'br\'')

    st.markdown("""
        #### Is your word in the vocabulary?
        """)

    user_word = st.text_input("Write your word")
    if user_word != '':
        st.write(f'Yeah! :white_check_mark:'
                 if user_word.lower() in counter_feature_names
                 else 'Nop! :no_entry_sign:')

    st.markdown("""
        #### How many positive/negative labels do we have?
        """)

    n_labels = (df.label.value_counts()
                .reset_index(drop=False)
                .rename(columns={'label': 'count',
                                 'index': 'label'}))
    st.plotly_chart(plot_labels(n_labels))

    st.write("The problem seems balanced! :sunglasses:")
