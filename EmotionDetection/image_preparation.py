"""Contains function for preparing and augmenting images"""
from collections import namedtuple

import cv2
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

from face_extract import excise_face

TrainableImage = namedtuple('TrainableImage', ['image_array', 'emotion'])

def rotate_image(array_image, rotation):
    """This function rotates the image clockwise or anticlockwise for data \
        augmentation"""
    rows, cols, channels = array_image.shape # pylint: disable=unused-variable

    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)

    return cv2.warpAffine(array_image, rotation_matrix, (cols, rows))

def prepare_image(file, emotion_code, reverse=False, rotate=0):
    """Load in single image, extract the face and denote the assigned emotion"""
    image = load_img(file, color_mode='grayscale', target_size=None)
    if reverse: image.transpose(Image.FLIP_LEFT_RIGHT)

    image_as_array = img_to_array(image)
    if rotate != 0: image_as_array = rotate_image(image_as_array, rotate)
    face_only = excise_face(image_as_array)

    return TrainableImage(face_only, emotion_code)
