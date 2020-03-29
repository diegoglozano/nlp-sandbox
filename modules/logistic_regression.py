import pandas as pd
import streamlit as st
import plotly.express as px
import random
import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from typing import Tuple

from utils.constants import SEED, N_JOBS
from utils.helpers import func_preprocessor, calculate_scores, fit_complete


def logistic_regression(df: pd.DataFrame, stop_words: Tuple[str]):
    st.title('NLP Sandbox')
    st.subheader('Steps')
    st.markdown("""

            1. Bag of Words     
            2. TF-iDF  
            3. Stemming  
            4. Stop Words  
            5. Classifier  
            6. Interpreting the model  
            7. Try the model!

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
                                coefs.nlargest(5)], axis=0).reset_index(drop=False).sort_values(by='importance',
                                                                                                ascending=True)

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
                    if probs[0, 0] > probs[0, 1]
                    else f'Probability of being positive: {round(probs[0, 1] * 100, 2)}%')
