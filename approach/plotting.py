import matplotlib.pyplot as plt


def plot(history, approach_name):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.5})

    axis[0].plot(epochs, acc, 'bo', label='Training accuracy')
    axis[0].plot(epochs, val_acc, 'b', label='Validation accuracy')
    axis[0].set_title(f'{approach_name} - training and validation accuracy')
    axis[0].legend()

    axis[1].plot(epochs, loss, 'bo', label='Training loss')
    axis[1].plot(epochs, val_loss, 'b', label='Validation loss')
    axis[1].set_title(f'{approach_name} - training and validation loss')
    axis[1].legend()

    plt.show()
