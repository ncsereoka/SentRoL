import numpy as np
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class Config:
    def __init__(self, x, y, embedding_dim, max_words, maxlen):
        # Dimension of the embedding layer
        self.embedding_dim = embedding_dim

        # How many words from the dataset to consider in total
        self.max_words = max_words

        # Used in padding, how long is a review
        self.maxlen = maxlen

        # All of training input (train + validation)
        self.x = x

        # All of training output (train + valdation)
        self.y = y

        # Word tokenizer, used for evaluation as well
        self.tokenizer = train_tokenizer(x, self.max_words)


def train_tokenizer(samples, max_words):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(samples)

    return tokenizer


def train(config):
    no_samples = len(config.x)
    no_validation_samples = no_samples // 5  # Validates on 20%, trains on 80%

    sequences = config.tokenizer.texts_to_sequences(config.x)
    print('Found %s unique tokens.' % len(config.tokenizer.word_index))

    data = pad_sequences(sequences, maxlen=config.maxlen)
    print('Shape of data tensor:', data.shape)

    labels = np.asarray(config.y)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    data = data[indices]
    labels = labels[indices]

    x_val = data[:no_validation_samples]
    x_train = data[no_validation_samples: no_samples]

    y_val = labels[:no_validation_samples]
    y_train = labels[no_validation_samples: no_samples]

    model = instantiate_model(config)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_data=(x_val, y_val))

    return model, history


def instantiate_model(config):
    model = Sequential()
    model.add(Embedding(config.max_words, config.embedding_dim, input_length=config.maxlen))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


def evaluate_model(model, config, x, y):
    sequences = config.tokenizer.texts_to_sequences(x)
    x = pad_sequences(sequences, maxlen=config.maxlen)
    y = np.asarray(y)

    loss, acc = model.evaluate(x, y)
    return loss, acc


def predict_sample(model, config, x, y, sample_index):
    print(f'Sample #{sample_index}:')
    print('Input: ' + x[sample_index])
    print('Expected output: ' + str(y[sample_index]))

    sequences = config.tokenizer.texts_to_sequences([x[sample_index]])
    i = pad_sequences(sequences, maxlen=config.maxlen)

    y = model.predict(i)
    print(f'Prediction: {y[0]}')
