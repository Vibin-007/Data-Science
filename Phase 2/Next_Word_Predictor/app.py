import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("Next Word Predictor")

text = st.text_area("Enter training text")

if st.button("Train Model"):

    text = text.lower()

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts([text])

    total_words = len(tokenizer.word_index) + 1

    input_sequences = []

    for line in text.split("."):

        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):

            seq = token_list[:i + 1]

            input_sequences.append(seq)

    max_len = max(len(x) for x in input_sequences)

    input_sequences = pad_sequences(
        input_sequences,
        maxlen=max_len,
        padding='pre'
    )

    X = input_sequences[:, :-1]

    y = input_sequences[:, -1]

    model = Sequential()

    model.add(Embedding(total_words, 32))

    model.add(LSTM(64))

    model.add(Dense(total_words, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    with st.spinner("Training model..."):

        model.fit(
            X,
            y,
            epochs=100,
            verbose=0
        )

    st.success("Model Trained Successfully!")

    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    st.session_state.max_len = max_len

if "model" in st.session_state:

    seed_text = st.text_input("Enter text for prediction")

    if st.button("Predict Next Word"):

        tokenizer = st.session_state.tokenizer
        model = st.session_state.model
        max_len = st.session_state.max_len

        token_list = tokenizer.texts_to_sequences(
            [seed_text.lower()]
        )[0]

        if len(token_list) == 0:

            st.error("Unknown word")

        else:

            token_list = pad_sequences(
                [token_list],
                maxlen=max_len - 1,
                padding='pre'
            )

            predicted = model.predict(
                token_list,
                verbose=0
            )

            predicted_word_index = np.argmax(predicted)

            output_word = ""

            for word, index in tokenizer.word_index.items():

                if index == predicted_word_index:

                    output_word = word
                    break

            st.success(f"Suggested Next Word: {output_word}")