import os, numpy, random
import hyper

from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from face_extract import excise_face
from cv2 import CascadeClassifier

import matplotlib.pyplot as pyplot

#test_img = load_img('example.png', color_mode='grayscale')

#print(type(test_img))

#img_array = img_to_array(test_img)

#print(img_array)
#print(img_array.shape)

#pyplot.figure()
#pyplot.imshow(img_array.reshape((490,640)), interpolation='none', cmap='gray')
#pyplot.show()

#WIDTH, HEIGHT = 640, 490
#SCALE_FACTOR = 2
#height_resize = HEIGHT/SCALE_FACTOR
#image_size = (int(HEIGHT/SCALE_FACTOR), int(WIDTH/SCALE_FACTOR))
image_size = (hyper.ROWS, hyper.COLS)

class ImageForCNN:
    def __init__(self, image_array, emotion):
        self.image_array = image_array
        self.emotion = emotion

def prepare_image_for_cnn(file, emotion_code, reverse=False):
    image = load_img(file, color_mode='grayscale', target_size=None)
    if reverse: image.transpose(Image.FLIP_LEFT_RIGHT)

    image_array = img_to_array(image)
    face_only = excise_face(image_array, 
        CascadeClassifier('haarcascade_frontalface_default.xml'))
    # print(face_only.shape)

    #return (image_array, emotion_code)
    return ImageForCNN(face_only, emotion_code)

def load_entire_emotion(directory, emotion_code):
    images_of_emotion = []

    for file in os.listdir(os.fsencode(directory)):
        filename = directory + '\\' + os.fsdecode(file)
        images_of_emotion.append(prepare_image_for_cnn(filename, emotion_code))
        # images_of_emotion.append(prepare_image_for_cnn(filename, emotion_code, True))

    return images_of_emotion

#images = []
#for file in os.listdir(os.fsencode("000 neutral")):
#    filename = "000 neutral\\" + os.fsdecode(file)
#    images.append(prepare_image_for_cnn(filename,0))

#for iii in range(5):
#    pyplot.figure(iii)
#    pyplot.imshow(images[iii].image_array.reshape((490,640)), interpolation='none', cmap='gray')
#pyplot.show()

#print(images[0].image_array.shape)

#pyplot.figure()
#pyplot.imshow(images[0].image_array.reshape((490,640)), interpolation='none', cmap='gray')
#pyplot.show()

# directories = [('000 neutral',0), 
#                ('001 surprise',1),
#                ('002 sadness',2),
#                ('003 fear',3),
#                ('004 anger',4),
#                ('005 disgust',5),
#                ('006 joy',6)]

directories = [('000 neutral',0)]

prepared_images = []

for directory in directories:
    prepared_images.extend(load_entire_emotion(directory[0],directory[1]))

# random.shuffle(prepared_images)

#pyplot.figure()
#pyplot.imshow(prepared_images[-1].image_array.reshape((490,640)), interpolation='none', cmap='gray')
#pyplot.show()

# Extract Trainable Data and Labels
dataset = []
#trainable_data = numpy.array()
indexed_labels = []
for prepared_image in prepared_images:
    #dataset.append(prepared_image.image_array)
    #image_array = prepared_image.image_array.reshape(490,640)
    #image_array = numpy.delete(image_array,2,1)
    image_array = numpy.asarray(prepared_image.image_array)
    #image_array = image_array.astype('float32')
    #image_array = numpy.expand_dims(prepared_image.image_array, 0)
    #image_array = numpy.expand_dims(image_array, 0)
    dataset.append(image_array.astype('float32'))
    indexed_labels.append(prepared_image.emotion)

#pyplot.figure()
#pyplot.imshow(dataset[42].reshape((490,640)), interpolation='none', cmap='gray')
#pyplot.show()

# Output Trainable Data as Array
print('Dataset Length: ' + str(len(dataset)))
print('Dataset Entry Shape:' + str(dataset[0].shape))
print('Dataset Entry Type: ' + str(dataset[0].dtype))
#print(dataset[0])
#dataset = numpy.asarray(dataset)
#dataset = numpy.stack(dataset)
#dataset = numpy.concatenate(dataset, axis=0)
#dataset = numpy.array(dataset)
#print(dataset.shape)
#print(dataset.dtype)
#print(dataset)
#trainable_data = numpy.array()
#dataset = numpy.asarray(dataset)
#trainable_data = numpy.stack(dataset)
#trainable_data = numpy.expand_dims(dataset[0],0)

#for image_array in dataset:
#    appending_array = numpy.expand_dims(image_array,0)
#    trainable_data = numpy.append(trainable_data, appending_array,0)
trainable_data = numpy.asarray(dataset)
trainable_data = numpy.expand_dims(trainable_data, -1)
print('Trainable Dataset Shape: ' + str(trainable_data.shape))
print('Trainable Dataset Type: ' + str(trainable_data.dtype))
#print(trainable_data)
numpy.save('ck_data', trainable_data)
print(' -- Trainable data saved --')

# Output Labels
indexed_labels = numpy.asarray(indexed_labels)
numpy.save('ck_labels', indexed_labels)
print(' -- Labels saved -- ')

# Check Output
#pyplot.figure()
#pyplot.imshow(trainable_data[0], interpolation='none', cmap='gray')
for iii in range(1):
    pyplot.figure(iii).suptitle(indexed_labels[iii])
    pyplot.imshow(trainable_data[iii].reshape((300,300)), interpolation='none', cmap='gray')
pyplot.show()