import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

from gensim.models import KeyedVectors

from typing import Tuple
from os import PathLike


def space():
    st.text("")


@st.cache
def read_data(input_path: PathLike,
              nrows: int) -> pd.DataFrame:
    return pd.read_csv(input_path,
                       nrows=nrows)


@st.cache
def read_stop_words(input_path: PathLike) -> Tuple[str]:
    return tuple(pd.read_csv(input_path, squeeze=True).values)


@st.cache(allow_output_mutation=True)
def read_word2vec(input_path: PathLike,
                  limit: int):
    word2vec = KeyedVectors.load_word2vec_format(input_path,
                                                 binary=True,
                                                 limit=limit)
    return word2vec


@st.cache
def get_sentences_len(serie: pd.Series) -> pd.Series:
    pattern_to_split = r'(?u)\b\w\w+\b'
    return serie.str.split(pattern_to_split).str.len()


@st.cache(hash_funcs={CountVectorizer: lambda _: None})
def get_counts(counter: CountVectorizer,
               text: pd.Series) -> (pd.DataFrame, list):
    counts_array = counter.fit_transform(text).toarray()
    counts_df = (pd.DataFrame(counts_array,
                              columns=counter.get_feature_names())
                 .sum().nlargest(20)
                 .reset_index(drop=False)
                 .rename(columns={0: 'total'}))
    return counts_df, counter.get_feature_names()


@st.cache
def plot_counter(counts_df: pd.DataFrame):
    return px.bar(counts_df, x='index', y='total')


@st.cache
def plot_labels(n_labels):
    return px.bar(n_labels, "label", "count")


@st.cache
def plot_word2vec(reducted_vectors: np.array,
                  vocab):
    return px.scatter(x=reducted_vectors[:, 0],
                      y=reducted_vectors[:, 1],
                      text=vocab)


@st.cache
def func_stemming(serie: pd.Series,
                  stemmer: nltk.stem.SnowballStemmer,
                  stop_words: bool = True,
                  stop_words_list: Tuple[str] = tuple()) -> pd.Series:
    splitted_serie = serie.str.split('(?u)\b\w\w+\b')
    stemmed_serie = splitted_serie.apply(lambda sent: [stemmer.stem(word)
                                                       for word in sent
                                                       if word.lower() not in stop_words_list])
    joined_serie = stemmed_serie.str.join(' ')
    return joined_serie


@st.cache
def func_clean_text(serie: pd.Series) -> pd.Series:
    return serie.str.replace('<br />', '')


@st.cache
def func_preprocessor(serie: pd.Series,
                      stemmer: nltk.stem.SnowballStemmer,
                      stemming: bool = True,
                      stop_words: bool = True,
                      clean_text: bool = True,
                      stop_words_list: Tuple[str] = tuple()) -> pd.Series:

    if stemming:
        stemmed_serie = func_stemming(serie, stemmer,
                                      stop_words=stop_words,
                                      stop_words_list=stop_words_list)

    else:
        stemmed_serie = serie

    if clean_text:
        cleaned_text = func_clean_text(stemmed_serie)
    else:
        cleaned_text = stemmed_serie

    return cleaned_text


@st.cache
def calculate_scores(pipeline: Pipeline,
                     data: pd.Series,
                     target: pd.Series,
                     folds: StratifiedKFold,
                     jobs: int) -> (float, float):
    scores = cross_val_score(pipeline,
                             data,
                             target,
                             cv=folds,
                             n_jobs=jobs)
    return scores.mean().round(2) * 100, scores.std().round(2) * 100


@st.cache(hash_funcs={Pipeline: lambda _: None})
def fit_complete(pipeline: Pipeline,
                 data: pd.Series,
                 target: pd.Series) -> Pipeline:
    return pipeline.fit(data, target)


@st.cache(allow_output_mutation=True)
def reduct_word2vec(pipeline: Pipeline,
                    data: np.array):
    return pipeline.fit_transform(data)
