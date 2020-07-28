"""Employ trained model on a single image to show prediction confidence"""
import argparse
import os
import numpy
import matplotlib.pyplot as pyplot
from matplotlib.ticker import MaxNLocator
import cv2
from hyper import FACE_SIZE

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
# pylint: disable=wrong-import-position
from face_extract import excise_face
from model_builders import reload_model

# def acquire_model(timestamp):
#     directory = hyper.MODEL_DIRECTORY
#     model_name = timestamp + '_model.json'
#     weights_name = timestamp + '_weights.h5'
    
#     with open(os.path.join(directory, model_name), 'r') as json_model:
#         model = model_from_json(json_model.read())
#     model.load_weights(os.path.join(directory, weights_name))

#     return model

# Command Line Arguments
parser = argparse.ArgumentParser(description='Demonstrate a single prediction \
                                 for the given model.')
parser.add_argument('-t', '--timestamp', help='The datetime for the model', 
                    required=True)
parser.add_argument('-i', '--image', help='Image to predict on', required=True)
args = parser.parse_args()

# Preprocess Image
full_image = cv2.imread(args.image)
gray_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
face_only = excise_face(gray_image)
face_only -= face_only.mean()
face_only /= face_only.std()
cropped_data = numpy.expand_dims(face_only, -1)
cropped_data = numpy.expand_dims(cropped_data, 0)
checkable_data = cropped_data.copy()

# Prepare Model
model = reload_model(args.timestamp)

# Make Prediction
prediction = model.predict(checkable_data).tolist()[0]
all_predictions = model.predict(checkable_data)
print('Predictions: ' + str(all_predictions))

# Show Image and Confidence Plot
figure, (axis_image, axis_confidence) = pyplot.subplots(1,2)
axis_image.imshow(checkable_data[0].reshape(FACE_SIZE), interpolation='none', 
                  cmap='gray')
axis_confidence.barh(numpy.arange(len(prediction)), width=prediction)
axis_confidence.yaxis.set_major_locator(MaxNLocator(integer=True))
axis_confidence.set_yticklabels(['',
                                 'neutral',
                                 'surprise',
                                 'sadness',
                                 'fear',
                                 'anger',
                                 'disgust',
                                 'joy'])
axis_confidence.set_xlim([0,1])
axis_confidence.xaxis.set_major_locator(MaxNLocator(10))
axis_confidence.grid(True, which='major', axis='x', linestyle='--')
pyplot.subplots_adjust(wspace=0.5)
pyplot.show()