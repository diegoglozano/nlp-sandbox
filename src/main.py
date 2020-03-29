import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import random
import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE

from gensim.models import KeyedVectors

from typing import Tuple
from pathlib import Path
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
    splitted_serie = serie.str.split('(?u)\b\w\w+\b')  # TODO: Cambiar el separador
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


# @st.cache
# @st.cache(hash_funcs={Pipeline: lambda pipe: (pipe.named_steps['preprocessor'], pipe.named_steps['vectorizer'])})
@st.cache(hash_funcs={Pipeline: lambda _: None})
def fit_complete(pipeline: Pipeline,
                 data: pd.Series,
                 target: pd.Series) -> Pipeline:
    return pipeline.fit(data, target)


@st.cache(allow_output_mutation=True)
def reduct_word2vec(pipeline: Pipeline,
                    data: np.array):
    return pipeline.fit_transform(data)


# ----------------------------------------------------------------------------------------------------------------------

ROOT_PATH = Path('..')
DATA_PATH = ROOT_PATH / 'data'
IMG_PATH = ROOT_PATH / 'img'

SEED = 42
N_JOBS = 1
NROWS = 2000
LIMITW2V = 1000

random.seed(SEED)

# ----------------------------------------------------------------------------------------------------------------------

st.balloons()

st.sidebar.title("NLP Sandbox")
st.sidebar.subheader("Choose your style :sunglasses:")
choice = st.sidebar.radio('',
                          ('Introduction', 'Logistic Regression', 'Deep Learning'))

df = read_data(DATA_PATH / 'train.csv', nrows=NROWS)
stop_words = read_stop_words(DATA_PATH / 'stop-words.txt')
word2vec = read_word2vec(DATA_PATH / 'GoogleNews-vectors-negative300.bin',
                         limit=LIMITW2V)

if choice == 'Introduction':
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

# ----------------------------------------------------------------------------------------------------------------------


if choice == 'Logistic Regression':

    st.title('NLP Sandbox')
    st.subheader('Steps')
    st.markdown("""
      
        [1. Bag of Words]()    
        [2. TF-iDF]()  
        [3. Stemming]()  
        [4. Stop Words]()  
        [5. Classifier]()  
        [6. Interpreting the model]()  
        [7. Try the model!]()
    
    """)

    # ------------------------------------------------------------------------------------------------------------------

    st.header('1. Bag of Words')

    example_sentences = pd.Series(['It is the best film I have ever seen',
                                   'The film is the worst',
                                   'This plot is going to be the best for the films',
                                   'The word cinema is going to be important cinema cinema cinema'])

    st.markdown(f"""
        First of all, we're going to encode our text in the simplest way: Bag of Words. It consists on counting 
        how many times a word appear in a sentence.  
        Let's say we want to encode the following sentences:
        - {example_sentences.iloc[0]}
        - {example_sentences.iloc[1]}
        - {example_sentences.iloc[2]}
        - {example_sentences.iloc[3]}
    """)

    bow = CountVectorizer()
    bow_text = bow.fit_transform(example_sentences)
    bow_df = pd.DataFrame(bow_text.toarray(),
                          columns=bow.get_feature_names())
    st.dataframe(bow_df)

    st.markdown("""
        Really? "the" is the most important word in the second/third sentence?
    """)

    # ----------------------------------------------------------------------------------------------------------------------

    st.header('2. TF-iDF')

    st.markdown("""
        Let's try to correct last problem. To achieve this, we're going to try TF-iDF:  
        _Term Frequency - Inverse Document Frequency_  
        It's calculated with the following formula:
    """)

    st.latex(r"tfidf = tf · \log{\frac{N}{df}}")

    st.markdown("""
        This way, a word that appears many times in just one sentence will be more important.
    """)

    tfidf = TfidfVectorizer()
    tfidf_text = tfidf.fit_transform(example_sentences)
    tfidf_df = pd.DataFrame(tfidf_text.toarray(),
                            columns=tfidf.get_feature_names())

    st.dataframe(tfidf_df.style.format("{:.2f}"))

    # ----------------------------------------------------------------------------------------------------------------------

    st.header('3. Stemming')

    st.markdown("""
        film / films are completely different words for the model. Can we solve this?  
    
        We are going to use stemming, a technique that transform a word into its root form:
    """)

    stemmer_example = nltk.stem.SnowballStemmer('english')
    preprocessor_example = FunctionTransformer(func_preprocessor,
                                               kw_args={'stemmer': stemmer_example,
                                                        'stemming': True,
                                                        'stop_words': True,
                                                        'clean_text': True,
                                                        'stop_words_list': []})

    joined_example_sentences = preprocessor_example.fit_transform(example_sentences)

    st.dataframe(pd.Series(joined_example_sentences,
                           name='text'))

    st.markdown("""
        Finally, film and films has the same meaning for our potential model. 
    """)

    st.markdown("""
    **Try it! Stem your own word!**
    """)

    stem_input = st.text_input('Write your word to stem:')
    if stem_input != '':
        st.write(f'{stem_input} --> {stemmer_example.stem(stem_input)}')

    # ----------------------------------------------------------------------------------------------------------------------

    st.header('4. Stop Words')

    st.markdown("""
        We can ignore some words and not taking them into account to train our model. For this purpose we have downloaded a
        list from internet. It contains, for example, some of the following words:
    """)

    sampled_stop_words = random.sample(stop_words, 5)
    tuple(st.markdown(f'- {word}') for word in sampled_stop_words)

    st.markdown('Try it! Is your word in the stopwords list?')

    stop_words_input = st.text_input('Check if your word is a stopword one:')
    if stop_words_input != '':
        st.markdown(f'*{stop_words_input}* is a stopword'
                    if stop_words_input in stop_words
                    else f'*{stop_words_input}* is not a stopword')

    # ----------------------------------------------------------------------------------------------------------------------

    st.header('5. Classifier')

    vectorizer = st.selectbox('What kind of vectorizer?',
                              ('Bag of Words', 'TF-iDF'))

    max_df = st.slider('Ignore vocabulary with higher frequency than...',
                       min_value=0.7,
                       max_value=1.0,
                       value=1.0)
    min_df = st.slider('Ignore vocabulary with lower frequency than...',
                       min_value=0.0,
                       max_value=0.3,
                       value=0.0)

    max_grams = st.slider('How many ngrams should we take?',
                          min_value=1,
                          max_value=10)

    preprocessing_options = ['Stemming',
                             'Stop Words',
                             'Clean text']

    preprocessing_selected = st.multiselect('Which preprocessing techniques should we use?',
                                            options=preprocessing_options)

    folds = st.slider('How many folds should we use for cross validation? They will be stratified, of course!',
                      min_value=2,
                      max_value=10,
                      value=2)

    stemmer = nltk.stem.SnowballStemmer('english')

    processor_args = {'stemmer': stemmer,
                      'stemming': 'Stemming' in preprocessing_selected,
                      'stop_words': 'Stop Words' in preprocessing_selected,
                      'clean_text': 'Clean text' in preprocessing_selected,
                      'stop_words_list': stop_words if 'Stop Words' in preprocessing_selected else []}

    preprocessor = FunctionTransformer(func_preprocessor,
                                       kw_args=processor_args)

    bow = CountVectorizer(ngram_range=(1, max_grams),
                          min_df=min_df,
                          max_df=max_df)
    tfidf = TfidfVectorizer(ngram_range=(1, max_grams),
                            min_df=min_df,
                            max_df=max_df)

    user_to_vectorizer = {'Bag of Words': bow,
                          'TF-iDF': tfidf}

    lr = LogisticRegression(random_state=SEED,
                            n_jobs=N_JOBS)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('vectorizer', user_to_vectorizer.get(vectorizer)),
        ('clf', lr)
    ])

    X = df.text
    y = df.label

    skf = StratifiedKFold(n_splits=folds,
                          shuffle=True,
                          random_state=SEED)

    mean_score, std_score = calculate_scores(pipeline,
                                             data=X,
                                             target=y,
                                             folds=skf,
                                             jobs=N_JOBS)

    # TODO: añadir otras métricas
    st.write(f'Logistic Regression accuracy in cross validation with {folds} folds is:')
    st.write(mean_score, '±', std_score, '%')

    # ----------------------------------------------------------------------------------------------------------------------

    st.header('6. Interpreting the model')

    st.markdown(f"""
    For this purpose we need to train the model.
    """)

    pipeline = fit_complete(pipeline,
                            X, y)

    coefs = pd.Series(pipeline.named_steps['clf'].coef_.ravel(),
                      index=pipeline.named_steps['vectorizer'].get_feature_names(),
                      name='importance')

    st.markdown("""
        Which are the most important words for a plot to be {positive|negative}?
    """)

    most_important = pd.concat([coefs.nsmallest(5),
                                coefs.nlargest(5)], axis=0).reset_index(drop=False).sort_values(by='importance', ascending=True)

    st.plotly_chart(px.bar(most_important, x='importance', y='index', orientation='h'))

    # ----------------------------------------------------------------------------------------------------------------------

    st.header('7. Try the model!')

    user_input = st.text_input("Write your own plot:")
    if user_input != '':
        array_input = pd.Series(user_input)
        prediction_input = pipeline.predict(array_input)
        st.markdown(':heart: Your plot is positive, you liked the film! :heart:'
                    if prediction_input == 1
                    else ':no_entry: Your plot is negative, you didn\'t like the film :no_entry:')
        probs = pipeline.predict_proba(array_input)
        st.markdown(f'Probability of being negative: {round(probs[0, 0] * 100, 2)}%'
                    if probs[0, 0]>probs[0, 1]
                    else f'Probability of being positive: {round(probs[0, 1] * 100, 2)}%')

        # TODO: Probar SHAP/ELI5 con esto
        # import shap
        # explainer = shap.LinearExplainer(pipeline.named_steps['clf'], pipeline.named_steps['vectorizer'].transform(df.text))
        # shap_values = explainer.shap_values(pipeline.named_steps['vectorizer'].transform(array_input))
        # shap_plot = shap.force_plot(explainer.expected_value, shap_values[0, :],
        #                             pipeline.named_steps['vectorizer'].transform(array_input).toarray(),
        #                             feature_names=pipeline.named_steps['vectorizer'].get_feature_names(),
        #                             matplotlib=True)
        # st.pyplot(shap_plot)


if choice == 'Deep Learning':

    st.title('NLP Sandbox')

    st.header('1. Word2Vec')

    st.markdown(f"""
        Let's give the problem a Deep Learning approach, making use of Word2Vec techniques.
    """)

    reductor = Pipeline([
        ('svd', TruncatedSVD(n_components=50, random_state=SEED)),
        ('tsne', TSNE(n_components=2, random_state=SEED))
    ])

    reducted_vectors = reduct_word2vec(reductor, word2vec.vectors)

    plotted_word2vec = plot_word2vec(reducted_vectors, list(word2vec.vocab))
    st.plotly_chart(plotted_word2vec)

    st.markdown('Try it! Get the most similar words:')
    word2vec_input = st.text_input('Write your word')
    if word2vec_input != '':
        try:
            most_similar_input = word2vec.most_similar(positive=word2vec_input)
            st.markdown(f"""
                - {most_similar_input[0][0]}
                - {most_similar_input[1][0]}
                - {most_similar_input[2][0]}
            """)
        except:
            st.markdown(f'{word2vec_input} is not in the Word2Vec vocab!')

    # st.header('2. Classification')

    # st.markdown(f"""
    #     To classify with Deep Learning techniques we will use Word2Vec, LSTMs and a fully connected layer.
    # """)

    # import tensorflow as tf
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Embedding, LSTM, Dense
    # from tensorflow.keras.preprocessing.text import Tokenizer
    # from tensorflow.keras.preprocessing.sequence import pad_sequences

    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(df.text)
    # sequences = tokenizer.texts_to_sequences(df.text)
    # padded_sequences = pad_sequences(sequences)

    # embedding_weights = np.vstack([np.zeros(word2vec.vectors.shape[1]),
    #                                word2vec.vectors])
    # vocab_size = len(word2vec.vocab) + 1
    # deep = Sequential([
    #     Embedding(input_dim=vocab_size,
    #               output_dim=300,
    #               weights=[embedding_weights],
    #               trainable=False,
    #               mask_zero=True),
    #     LSTM(128),
    #     Dense(1, activation='sigmoid')
    # ])
    # deep.compile(loss='binary_crossentropy', optimizer='adam')
    # st.text(sequences)
    # deep.fit(sequences, df.label.values, epochs=1)



