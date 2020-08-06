"""Contains functions for recording various metrics of trained models"""
import os
from itertools import product

import matplotlib.pyplot as pyplot
from matplotlib.ticker import MaxNLocator, PercentFormatter, MultipleLocator
import numpy
from sklearn.metrics import confusion_matrix

from hyper import BATCH_SIZE, MODEL_DIRECTORY, SCALE_DOWN_FACTOR

labels = ['joy', 'disgust', 'anger', 'fear', 'sadness', 'surprise', 'neutral']

def author_confusion_matrix(true, predicted, timestamp):
    """Prepare and save confusion matrix for tested model"""
    matrix_filename = timestamp + '_confusion.png'
    matrix = confusion_matrix(true, predicted, normalize='true')

    figure, axis = pyplot.subplots()
    matrix_image = axis.imshow(matrix, interpolation='nearest',
                               cmap=pyplot.cm.get_cmap(name='Blues'))
    axis.set(title='Confusion Matrix')
    figure.colorbar(matrix_image)
    tick_marks = numpy.arange(len(labels))
    axis.set_xticks(tick_marks)
    axis.set_xticklabels(labels, rotation=45, ha='right')
    axis.set_yticks(tick_marks)
    axis.set_yticklabels(labels)
    threshold = matrix.max()/2

    for iii, jjj in product(range(matrix.shape[0]), range(matrix.shape[1])):
        axis.text(jjj, iii, format(matrix[iii, jjj], '0.2f'),
                  horizontalalignment='center',
                  color='white' if matrix[iii, jjj] > threshold else 'black')

    axis.set_ylabel('True Label')
    axis.set_xlabel('Predicted Label')
    figure.tight_layout()

    figure.savefig(os.path.join(MODEL_DIRECTORY, matrix_filename))

def log_details(timestamp, history, epochs, model):
    """Save the pertinent details of a trained model in a .txt file"""
    filename = timestamp + '_details.txt'

    with open(os.path.join(MODEL_DIRECTORY, filename), 'w') as file:
        file.write('Model Used: ' + model + '\n')
        file.write('Training Accuracy: ' + '{:.3f}'.format(history['acc'][-1]) + '\n')
        file.write('Validation Accuracy: ' + '{:.3f}'.format(history['val_acc'][-1]) + '\n')
        file.write('Note the above metrics may not represent the checkpointed model\n')
        file.write('Scale Down Factor: ' + str(SCALE_DOWN_FACTOR) + '\n')
        file.write('Batch Size: ' + str(BATCH_SIZE) + '\n')
        file.write('No. of Epochs (Completed/Requested): ' +
                   str(len(history['acc'])) + '/' + str(epochs) + '\n')

def plot_training(plotting_history, plot_filename):
    """Plot and export the history of the changes in the loss and the accuracy \
        for both the training and validation datasets"""
    figure, (axis_loss, axis_accuracy) = pyplot.subplots(2, 1, sharex=True)
    pyplot.subplots_adjust(hspace=0.01)

    axis_loss.plot(plotting_history['loss'], label='training', color='b')
    axis_loss.plot(plotting_history['val_loss'], label='validation', color='r')
    axis_loss.set(ylabel='Loss')
    axis_loss.grid(True, which='major', axis='x', linewidth='0.5')
    axis_loss.legend()

    axis_accuracy.plot(plotting_history['acc'], label='training', color='b')
    axis_accuracy.plot(plotting_history['val_acc'], label='validation', color='r')
    axis_accuracy.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis_accuracy.yaxis.set_major_locator(MultipleLocator(base=0.25))
    axis_accuracy.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis_accuracy.set_ylim([0, 1.05])
    axis_accuracy.set(xlabel='Epoch', ylabel='Accuracy(%)')

    figure.savefig(plot_filename)
    