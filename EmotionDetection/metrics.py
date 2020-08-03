"""Contains functions for recording various metrics of trained models"""
import os
from itertools import product

import matplotlib.pyplot as pyplot
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
    axis.set_xticklabels(labels)
    axis.set_xticks(tick_marks)
    axis.set_xticklabels(labels)
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
