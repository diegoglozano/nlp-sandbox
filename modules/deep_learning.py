import streamlit as st

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE

from utils.constants import SEED
from utils.helpers import reduct_word2vec, plot_word2vec


def deep_learning(word2vec):
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
