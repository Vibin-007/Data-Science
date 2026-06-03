import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "I love this movie",
    "This movie is amazing",
    "Excellent acting",
    "I hate this movie",
    "This movie is terrible",
    "Worst film ever"
]

labels = [1, 1, 1, 0, 0, 0]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

max_length = 5
padded_sequences = pad_sequences(
    sequences,
    maxlen=max_length,
    padding='post'
)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=16,
        input_length=max_length
    ),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(
    padded_sequences,
    labels,
    epochs=50,
    verbose=1
)

test_text = ["I really love this film"]

test_seq = tokenizer.texts_to_sequences(test_text)
test_pad = pad_sequences(
    test_seq,
    maxlen=max_length,
    padding='post'
)

prediction = model.predict(test_pad)

if prediction[0][0] > 0.5:
    print("Positive")
else:
    print("Negative")

print("Probability:", prediction[0][0])