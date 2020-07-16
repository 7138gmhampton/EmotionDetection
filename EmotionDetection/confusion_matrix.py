import numpy, itertools
import matplotlib.pyplot as pyplot

from sklearn.metrics import confusion_matrix

labels = ['joy','disgust','anger','fear','sadness','surprise','neutral']

# Prepare Matrix
predicted_list = numpy.load('prediction_list.npy')
true_list = numpy.load('true_list.npy')

model_matrix = confusion_matrix(true_list, predicted_list, normalize='all')
print(model_matrix)

# Display Graphic
pyplot.imshow(model_matrix, interpolation='nearest', cmap=pyplot.cm.Blues)
pyplot.title('Confusion Matrix')
pyplot.colorbar()
tick_marks = numpy.arange(len(labels))
pyplot.xticks(tick_marks, labels, rotation=45)
pyplot.yticks(tick_marks, labels)
threshold = model_matrix.max() / 2

for iii, jjj in itertools.product(range(model_matrix.shape[0]), range(model_matrix.shape[1])):
    pyplot.text(jjj, iii, format(model_matrix[iii,jjj], '0.2f'), horizontalalignment='center', 
                color='white' if model_matrix[iii,jjj] > threshold else 'black')

pyplot.ylabel('True label')
pyplot.xlabel('Predicted label')
pyplot.tight_layout()

pyplot.show()