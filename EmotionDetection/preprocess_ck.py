"""Prepare the Cohn-Kanade Dataset for Training"""
import argparse
import os
import random

import matplotlib.pyplot as pyplot
import numpy

from hyper import FACE_SIZE
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
os.environ['PLAIDML_VERBOSE'] = '4'
# pylint: disable=wrong-import-position
from image_preparation import prepare_image
from progress_bar import progress_bar

# Command Line Parameters
parser = argparse.ArgumentParser(description='Prepare and augment the Cohn-Kanade\
                                  for training, validation and testing')
parser.add_argument('-r', '--restrict', action='store_true',
                    help='Preprocess without augmenting')
args = parser.parse_args()

CLOCKWISE = 5
ANTICLOCKWISE = -CLOCKWISE

def load_entire_emotion(directory_of_images, emotion_code):
    """Prepare an entire emotion's worth of images for training"""
    images = []

    # for file in os.listdir(os.fsencode(directory_of_images)):
    for file in progress_bar(os.listdir(os.fsencode(directory_of_images)),
                             suffix=directory_of_images):
        filename = directory_of_images + '\\' + os.fsdecode(file)
        images.append(prepare_image(filename, emotion_code))
        if not args.restrict:
            images.append(prepare_image(filename, emotion_code, rotate=CLOCKWISE))
            images.append(prepare_image(filename, emotion_code, rotate=ANTICLOCKWISE))
            images.append(prepare_image(filename, emotion_code, True))
            images.append(prepare_image(filename, emotion_code, True, rotate=CLOCKWISE))
            images.append(prepare_image(filename, emotion_code, True, rotate=ANTICLOCKWISE))

    return images

directories = [('000 neutral', 0),
               ('001 surprise', 1),
               ('002 sadness', 2),
               ('003 fear', 3),
               ('004 anger', 4),
               ('005 disgust', 5),
               ('006 joy', 6)]

prepared_images = []

print(' -- Prepare and Augment --')
for directory in directories:
    prepared_images.extend(load_entire_emotion(directory[0], directory[1]))

random.shuffle(prepared_images)

# Extract Trainable Data and Labels
dataset = []
indexed_labels = []
for prepared_image in prepared_images:
    image_array = numpy.asarray(prepared_image.image_array)
    dataset.append(image_array.astype('float32'))
    indexed_labels.append(prepared_image.emotion)

# Output Trainable Data as Array
print('Dataset Length: ' + str(len(dataset)))
print('Dataset Entry Shape:' + str(dataset[0].shape))
print('Dataset Entry Type: ' + str(dataset[0].dtype))

trainable_data = numpy.asarray(dataset)
trainable_data = numpy.expand_dims(trainable_data, -1)
print('Trainable Dataset Shape: ' + str(trainable_data.shape))
print('Trainable Dataset Type: ' + str(trainable_data.dtype))
numpy.save('ck_data', trainable_data)
print(' -- Trainable data saved --')

# Output Labels
indexed_labels = numpy.asarray(indexed_labels)
numpy.save('ck_labels', indexed_labels)
print(' -- Labels saved -- ')

# Check Output
for iii in range(3):
    pyplot.figure(iii).suptitle(indexed_labels[iii])
    pyplot.imshow(trainable_data[iii].reshape(FACE_SIZE), interpolation='none',
                  cmap='gray')
pyplot.show()
