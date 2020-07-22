import numpy, cv2

from mtcnn.mtcnn import MTCNN
from keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot
from cv2 import CascadeClassifier

# print(cv2.__version__)

def excise_face(image_array, classifier):
    int_array = image_array.astype(numpy.uint8)
    faces = classifier.detectMultiScale(int_array, 1.3, 0)

    x_start, y_start, width, height = faces[0]
    x_end, y_end = x_start + width, y_start + height
    output = int_array[y_start:y_end, x_start:x_end]
    output = cv2.resize(output, (300,300))

    return output.astype(numpy.float32)

# Load an Image
# raw_image = load_img('example3.png', color_mode='grayscale')
# raw_image = img_to_array(raw_image).squeeze()
# raw_image = raw_image.astype(numpy.uint8)
# raw_image = raw_image.astype(numpy.uint8)
# print(raw_image.shape)
# print(raw_image)
# # other_image = pyplot.imread('example.png')
# # print(other_image.shape)
# other_image = cv2.imread('example.png')
# gray_image = cv2.cvtColor(other_image, cv2.COLOR_BGR2GRAY)
# print(gray_image.shape)
# print(gray_image)

# # Extract Face
# # detector = MTCNN()
# # results = detector.detect_faces(other_image)
# # print(results)
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# detected_faces = face_cascade.detectMultiScale(raw_image, 1.3, 1)
# # print(faces[0])
# # print(len(faces))

# # Output Result as Array
# x_start, y_start, width, height = detected_faces[0]
# output_image = raw_image[y_start:(y_start+height), x_start:(x_start+width)]
# output_image = cv2.resize(output_image, (300,300))
# print(output_image.shape)

# # Display Result
# # output_image = output_image.astype(numpy.float32)
# # cv2.imshow('output', output_image)
# # pyplot.figure()
# # pyplot.imshow(output_image, interpolation='none', cmap='gray')
# # pyplot.show()
# # output_image = output_image.astype(numpy.float32)
# # print(output_image)
# # cv2.waitKey()
# # cv2.destroyAllWindows()

# first_image = load_img('example.png', color_mode='grayscale')
# first_image = img_to_array(first_image).squeeze()
# second_image = load_img('example2.png', color_mode='grayscale')
# second_image = img_to_array(second_image).squeeze()
# third_image = load_img('example3.png', color_mode='grayscale')
# third_image = img_to_array(third_image).squeeze()

# # first_output = excise_face(first_image, face_cascade)
# # second_output = excise_face(second_image, face_cascade)
# # third_output = excise_face(third_image, face_cascade)

# input_images  = [first_image, second_image, third_image]

# for full_image in input_images:
#     face_only = excise_face(full_image, face_cascade)

#     figure, (before, after) = pyplot.subplots(1,2)
#     before.imshow(full_image, interpolation='none', cmap='gray')
#     after.imshow(face_only, interpolation='none', cmap='gray')

# pyplot.show()