"""Prepare the Cohn-Kanade Dataset for Training"""
import os
import random
from collections import namedtuple
import numpy
import cv2
from PIL import Image
import matplotlib.pyplot as pyplot
from progress_bar import progress_bar

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
# pylint: disable=wrong-import-position
from keras.preprocessing.image import load_img, img_to_array
from face_extract import excise_face
from hyper import FACE_SIZE

TrainableImage = namedtuple('TrainableImage', ['image_array', 'emotion'])

CLOCKWISE = 5
ANTICLOCKWISE = -CLOCKWISE

def rotate_image(array_image, rotation):
    """This function rotates the image clockwise or anticlockwise for data \
        augmentation"""
    rows, cols, channels = array_image.shape # pylint: disable=unused-variable
    
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
    
    return cv2.warpAffine(array_image, rotation_matrix, (cols, rows))

def prepare_image_for_cnn(file, emotion_code, reverse=False, rotate=0):
    """Load in single image, extract the face and denote the assigned emotion"""
    image = load_img(file, color_mode='grayscale', target_size=None)
    if reverse: image.transpose(Image.FLIP_LEFT_RIGHT)

    image_as_array = img_to_array(image)
    if rotate != 0: image_as_array = rotate_image(image_as_array, rotate)
    face_only = excise_face(image_as_array)

    return TrainableImage(face_only, emotion_code)

def load_entire_emotion(directory_of_images, emotion_code):
    """Prepare an entire emotion's worth of images for training"""
    images_of_emotion = []

    # for file in os.listdir(os.fsencode(directory_of_images)):
    for file in progress_bar(os.listdir(os.fsencode(directory_of_images)), suffix=directory_of_images):
        filename = directory_of_images + '\\' + os.fsdecode(file)
        images_of_emotion.append(prepare_image_for_cnn(filename, emotion_code))
        images_of_emotion.append(prepare_image_for_cnn(filename, emotion_code, rotate=CLOCKWISE))
        images_of_emotion.append(prepare_image_for_cnn(filename, emotion_code, True))

    return images_of_emotion

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
    pyplot.imshow(trainable_data[iii].reshape(FACE_SIZE), interpolation='none',\
        cmap='gray')
pyplot.show()
