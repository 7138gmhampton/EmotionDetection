"""Train a model on the preprocessed Cohn-Kanade Dataset and save that model"""
import os
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
# pylint: disable=wrong-import-position
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from model_builders import prepare_model
from hyper import BATCH_SIZE, SCALE_DOWN_FACTOR, DROPOUT

# Command Line Parameters
parser = argparse.ArgumentParser(description='Train CNN model with Cohn-Kanade dataset.')
parser.add_argument('-e', '--epochs', type=int, required=True)
args = parser.parse_args()

# Hyperparameters
# no_of_features = hyper.NO_OF_FEATURES
# no_of_labels = hyper.NO_OF_LABELS
# batch_size = hyper.BATCH_SIZE
# no_of_epochs = args.epochs
# rows, cols = hyper.ROWS, hyper.COLS

def plot_training(plotting_history, plot_filename):
    """Plot and export the history of the changes in the loss and the accuracy \
        for both the training and validation datasets"""
    figure, (axis_loss, axis_accuracy) = pyplot.subplots(2, 1, sharex=True)
    pyplot.subplots_adjust(hspace=0.01)

    axis_loss.plot(plotting_history['loss'], label='training')
    axis_loss.plot(plotting_history['val_loss'], label='validation')
    axis_loss.set(ylabel='Loss')

    axis_accuracy.plot(plotting_history['acc'], label='training')
    axis_accuracy.plot(plotting_history['val_acc'], label='validation')
    axis_accuracy.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis_accuracy.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis_accuracy.set(xlabel='Epoch', ylabel='Accuracy(%)')

    figure.savefig(plot_filename)

def save_trained_model(trained_model, training_history):
    """Save the model and it weights. Also output a text file detailing the \
        hyperparameters and a graphic of the training history"""
    directory = 'models'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    model_name = timestamp + '_model.json'
    weights_name = timestamp + '_weights.h5'
    details_name = timestamp + '_details.txt'
    graph_name = timestamp + '_training.png'

    model_json = trained_model.to_json()
    with open(os.path.join(directory, model_name), 'w') as json_file:
        json_file.write(model_json)
    with open(os.path.join(directory, details_name), 'w') as text_file:
        text_file.write('Training Accuracy: ' +
                        '{:1.3f}'.format(training_history['acc'][-1]) + '\n')
        text_file.write('Validation Accuracy: ' +
                        '{:1.3f}'.format(training_history['val_acc'][-1]) + '\n')
        text_file.write('Scale Down Factor: ' +
                        '{:2d}'.format(SCALE_DOWN_FACTOR) + '\n')
        text_file.write('Batch Size: ' + '{:3d}'.format(BATCH_SIZE) + '\n')
        text_file.write('No. of Epochs: ' + '{:3d}'.format(args.epochs) + '\n')
        text_file.write('Dropout: ' + '{:1.3f}'.format(DROPOUT) + '\n')
    trained_model.save_weights(os.path.join(directory, weights_name))
    plot_training(training_history, os.path.join(directory, graph_name))

    print(' -- Model Saved --')

# Load Training Data
data = numpy.load('ck_data.npy')
labels = numpy.load('ck_labels.npy')

# Standardise
data -= numpy.mean(data, 0)
data /= numpy.std(data, 0)

# Change Labels to Categorical
labels = to_categorical(labels)

# Section Data
data_train, data_test, labels_train, labels_test = \
    train_test_split(data, labels, test_size=0.1, random_state=42)
data_train, data_valid, labels_train, labels_valid = \
    train_test_split(data_train, labels_train, test_size=0.1, random_state=42)

# Save Test Data
numpy.save('ck_test_data', data_test)
numpy.save('ck_test_labels', labels_test)
print(' -- Test Data Saved --')

# Train Model
# build = build_shanks
model = prepare_model('shanks')
early_stopper = EarlyStopping(monitor='val_loss', mode='max', verbose=1,
                              patience=20, min_delta=0.001)
rate_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
training = model.fit(numpy.array(data_train),
                     numpy.array(labels_train),
                     batch_size=BATCH_SIZE,
                     epochs=args.epochs,
                     verbose=1,
                     validation_data=(numpy.array(data_valid),
                                      numpy.array(labels_valid)),
                     shuffle=True,
                     callbacks=[rate_reducer, early_stopper])

# Save Model
save_trained_model(model, training.history)
