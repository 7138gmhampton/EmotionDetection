"""Contains functions for recording various metrics of trained models"""
import os
from itertools import product

import matplotlib.pyplot as pyplot
from matplotlib.ticker import MaxNLocator, PercentFormatter, MultipleLocator
import numpy
from sklearn.metrics import confusion_matrix

from hyper import BATCH_SIZE, MODEL_DIRECTORY, SCALE_DOWN_FACTOR

labels = ['joy', 'disgust', 'anger', 'fear', 'sadness', 'surprise', 'neutral']

def _get_loss_limits(history):
    training_max = max(history['loss'])
    validation_max = max(history['val_loss'])
    highest = training_max if training_max > validation_max else validation_max

    padding = 0.05 * highest

    return [0-padding, highest+padding]

def _prepare_graph(graph, history, loss):
    metric = ('loss', 'val_loss', 'Loss') if loss else ('acc', 'val_acc', 'Accuracy (%)')

    graph.plot(history[metric[0]], label='training', color='b')
    graph.plot(history[metric[1]], label='validation', color='r')
    graph.set_ylabel(ylabel=metric[2])
    graph.xaxis.grid(True, which='major', linewidth='0.5')
    graph.yaxis.grid(True, which='major', linestyle=':', linewidth='0.75')

    if loss:
        graph.yaxis.set_major_locator(MaxNLocator(nbins=5))
        graph.set_ylim(_get_loss_limits(history))
        graph.legend()
    else:
        graph.xaxis.set_major_locator(MaxNLocator(integer=True))
        graph.yaxis.set_major_locator(MultipleLocator(base=0.20))
        graph.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        graph.set_ylim([0, 1.05])
        graph.set_xlabel('Epoch')

def _colourise_text(graph, matrix, threshold):
    for iii, jjj in product(range(matrix.shape[0]), range(matrix.shape[1])):
        graph.text(jjj, iii, format(matrix[iii, jjj], '0.2f'),
                   horizontalalignment='center',
                   color='white' if matrix[iii, jjj] > threshold else 'black')

def author_confusion_matrix(true, predicted, timestamp):
    """Prepare and save confusion matrix for tested model"""
    matrix_filename = timestamp + '_confusion.png'
    matrix = confusion_matrix(true, predicted, normalize='true')
    figure, axis = pyplot.subplots()

    matrix_image = axis.imshow(matrix, interpolation='nearest',
                               cmap=pyplot.cm.get_cmap(name='Blues'))
    axis.set(title='Confusion Matrix')
    tick_marks = numpy.arange(len(labels))
    axis.set_xticks(tick_marks)
    axis.set_xticklabels(labels, rotation=45, ha='right')
    axis.set_yticks(tick_marks)
    axis.set_yticklabels(labels)
    axis.set_ylabel('True Label')
    axis.set_xlabel('Predicted Label')
    _colourise_text(axis, matrix, matrix.max()/2)

    figure.colorbar(matrix_image)
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
    figure, (axis_loss, axis_accuracy) = pyplot.subplots(2, 1, sharex=True, figsize=(8, 8))
    pyplot.subplots_adjust(hspace=0.00)

    _prepare_graph(axis_loss, plotting_history, True)
    _prepare_graph(axis_accuracy, plotting_history, False)

    figure.savefig(plot_filename)
