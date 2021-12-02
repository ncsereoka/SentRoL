import numpy as np
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences


class Config:
    def __init__(self, embedding_dim, max_words, max_len):
        # Dimension of the embedding layer
        self.embedding_dim = embedding_dim

        # How many words from the dataset to consider in total
        self.max_words = max_words

        self.max_len = max_len


def train(config, tokenizer, x, y):
    sequences = tokenizer.texts_to_sequences(x)
    print('Found %s unique tokens.' % len(tokenizer.word_index))

    x = pad_sequences(sequences, maxlen=config.max_len)
    print('Shape of x:', x.shape)

    y = np.asarray(y)
    print('Shape of y:', y.shape)

    no_samples = x.shape[0]
    no_validation_samples = no_samples // 5  # Validates on 20%, trains on 80%

    x_val = x[:no_validation_samples]
    x_train = x[no_validation_samples: no_samples]

    y_val = y[:no_validation_samples]
    y_train = y[no_validation_samples: no_samples]

    model = instantiate_model(config, x_train.shape[1])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    print('\n\nTraining the model...\n')
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_data=(x_val, y_val))

    return model, history


def instantiate_model(config, input_length):
    model = Sequential()
    model.add(Embedding(config.max_words, config.embedding_dim, input_length=input_length))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print('\n\nPreparing to train the following model:\n')
    print(model.summary())

    return model


def evaluate_model(model, config, tokenizer, x, y):
    sequences = tokenizer.texts_to_sequences(x)
    x = pad_sequences(sequences, maxlen=config.max_len)
    y = np.asarray(y)

    loss, acc = model.evaluate(x, y)
    return loss, acc


def predict_sample(model, config, tokenizer, x, y, sample_index):
    print(f'Sample #{sample_index}:')
    print('Input: ' + x[sample_index])
    print('Expected output: ' + str(y[sample_index]))

    sequences = tokenizer.texts_to_sequences([x[sample_index]])
    i = pad_sequences(sequences, maxlen=config.max_len)

    y = model.predict(i)
    print(f'Prediction: {y[0]}')
