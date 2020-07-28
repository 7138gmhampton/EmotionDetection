"""Employs OpenCV API to extract faces from given images"""
import numpy
import cv2
from hyper import FACE_SIZE, FACE_EXTRACTOR

def display_face(source_image, face, index):
    """Display an excise \'face\'"""
    x_start, y_start, width, height = face
    x_end, y_end = x_start + width, y_start + height

    display = source_image[y_start:y_end, x_start:x_end]

    cv2.imshow('multiface'+ str(index), display)

def excise_face(image_array):
    """Detect face and crop image to that alone"""
    classifier = cv2.CascadeClassifier(FACE_EXTRACTOR)
    int_array = image_array.astype(numpy.uint8)
    faces = classifier.detectMultiScale(int_array, 1.1, 5, minSize=(200, 200),\
        maxSize=(350, 350))
    if len(faces) != 1:
        print('No of faces detected: ' + str(len(faces)))
    if len(faces) > 1:
        display_face(int_array, faces[0], 0)
        display_face(int_array, faces[1], 1)
        cv2.waitKey()

    x_start, y_start, width, height = faces[0]
    x_end, y_end = x_start + width, y_start + height
    output = int_array[y_start:y_end, x_start:x_end]
    output = cv2.resize(output, FACE_SIZE)

    return output.astype(numpy.float32)
