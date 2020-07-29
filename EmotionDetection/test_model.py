"""
Do a prediction across the test dataset and compare to assigned emotions by 
determining accuracy and producing confusion matrix
"""
import argparse
import os
import numpy

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
# pylint: disable=wrong-import-position
from keras.models import model_from_json
from metrics import author_confusion_matrix
from hyper import MODEL_DIRECTORY
from model_builders import reload_model

# def acquire_model(timestamp):
#     directory = MODEL_DIRECTORY
#     model_name = timestamp + '_model.json'
#     weights_name = timestamp + '_weights.h5'
    
#     with open(os.path.join(directory, model_name), 'r') as json_model:
#         #return model_from_json(json_model.read())
#         model = model_from_json(json_model.read())
#     model.load_weights(os.path.join(directory, weights_name))

#     return model

def log_accuracy(timestamp, accuracy):
    details_name = timestamp + '_details.txt'

    with open(os.path.join(MODEL_DIRECTORY, details_name), 'r+') as details:
        if 'Test Accuracy: ' not in details.read():
            details.write('Test Accuracy: ' + '{:1.3f}'.format(accuracy) + '\n')

# Set up Command Line Arguments
parser = argparse.ArgumentParser(description='Apply test examples to trained model.')
parser.add_argument('-t', '--timestamp', help='The datetime for the model', required=True)
args = parser.parse_args()

#print(args.model)

# Prepare Model
# timestamp = args.timestamp

# model = acquire_model(args.timestamp)
model = reload_model(args.timestamp)
print(' -- Model loaded from file --')

# Load Testing Data
data = numpy.load('ck_test_data.npy')
labels = numpy.load('ck_test_labels.npy')

# Predict Based on Model and Test Data
predicted_output = model.predict(data).tolist()
# print(predicted_output)
true_output = labels.tolist()

# Compare Prediction to Truth
count_of_matches = 0
prediction_list = []
true_list = []

for iii in range(len(labels)):
    predicted_emotion = max(predicted_output[iii])
    true_emotion = max(true_output[iii])
    prediction_list.append(predicted_output[iii].index(predicted_emotion))
    true_list.append(true_output[iii].index(true_emotion))
    if predicted_output[iii].index(predicted_emotion) == true_output[iii].index(true_emotion):
        count_of_matches += 1

#print((count_of_matches/len(labels))*100)
accuracy = (count_of_matches/len(labels))
print('Accuracy of ' + args.timestamp +' model: ' + '{:2.2f}'.format(accuracy*100) + '%')
log_accuracy(args.timestamp, accuracy)

# Save Prediction and True Labels Lists for Confusion Matrix
numpy.save('prediction_list', prediction_list)
numpy.save('true_list', true_list)
print(' -- Lists saved --')

author_confusion_matrix(true_list, prediction_list, args.timestamp)