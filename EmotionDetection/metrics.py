import os
import numpy
import itertools
from itertools import product
import matplotlib.pyplot as pyplot
from sklearn.metrics import confusion_matrix
from hyper import MODEL_DIRECTORY

labels = ['joy', 'disgust', 'anger', 'fear', 'sadness', 'surprise', 'neutral']

def author_confusion_matrix(true, predicted, timestamp):
    matrix_filename = timestamp + '_confusion.png'
    matrix = confusion_matrix(true, predicted, normalize='true')
    
    # figure = pyplot.figure()
    figure, axis = pyplot.subplots()
    matrix_image = axis.imshow(matrix, interpolation='nearest', cmap=pyplot.cm.Blues)
    # axis.title('Confusion Matrix')
    axis.set(title='Confusion Matrix')
    figure.colorbar(matrix_image)
    tick_marks = numpy.arange(len(labels))
    # axis.xticks(tick_marks, labels, rotation=45)
    axis.set_xticks(tick_marks)
    axis.set_xticklabels(labels)
    # axis.yticks(tick_marks, labels)
    axis.set_xticks(tick_marks)
    axis.set_xticklabels(labels)
    threshold = matrix.max()/2
    
    for iii, jjj in product(range(matrix.shape[0]), range(matrix.shape[1])):
        axis.text(jjj, iii, format(matrix[iii, jjj], '0.2f'), horizontalalignment='center', color='white' if matrix[iii,jjj] > threshold else 'black')
        
    # axis.ylabel('True Label')
    axis.set_ylabel('True Label')
    # axis.xlabel('Predicted Label')
    axis.set_xlabel('Predicted Label')
    figure.tight_layout()
    
    figure.savefig(os.path.join(MODEL_DIRECTORY, matrix_filename))