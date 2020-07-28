import argparse, hyper, numpy, os
import matplotlib.pyplot as pyplot
from hyper import FACE_SIZE
import cv2
from cv2 import CascadeClassifier, normalize

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras.preprocessing.image import load_img, img_to_array
from keras.models import model_from_json
from face_extract import excise_face
from matplotlib.ticker import MaxNLocator

def acquire_model(timestamp):
    directory = hyper.MODEL_DIRECTORY
    model_name = timestamp + '_model.json'
    weights_name = timestamp + '_weights.h5'
    
    with open(os.path.join(directory, model_name), 'r') as json_model:
        model = model_from_json(json_model.read())
    model.load_weights(os.path.join(directory, weights_name))

    return model

# Command Line Arguments
parser = argparse.ArgumentParser(description='Demonstrate a single prediction for the given model.')
parser.add_argument('-t', '--timestamp', help='The datetime for the model', required=True)
parser.add_argument('-i', '--image', help='Image to predict on', required=True)
args = parser.parse_args()

# Preprocess Image
# image_size = (hyper.ROWS, hyper.COLS)
# image_to_check = load_img(args.image, color_mode='grayscale')
# image_array = img_to_array(image_to_check)
# face_only = excise_face(image_array, CascadeClassifier('haarcascade_frontalface_default.xml'))
# print(face_only.shape)
# image_array = numpy.asarray(image_array).squeeze()
# face_only = numpy.asarray(face_only).squeeze()
# print(face_only.shape)
# cropped_image = face_only.copy()
# normalize(face_only.astype(numpy.uint8), face_only, alpha=0, beta=255, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
# data_wrapper = [image_array.astype('float32')]
# data_wrapper = [face_only.astype('float32')]
# checkable_data = numpy.asarray(data_wrapper)
# checkable_data = numpy.expand_dims(checkable_data, -1)
# print(checkable_data.shape)
# print(checkable_data)
# image = load_img(args.image, color_mode='grayscale')
# image_as_array = img_to_array(image)
# face_only = excise_face(image_as_array, CascadeClassifier('haarcascade_frontalface_default.xml'))
# print(face_only.shape)
# print(face_only)
# data_wrapper = []
# data_wrapper.append(face_only)
# checkable_data = numpy.asarray(data_wrapper)
# checkable_data = numpy.expand_dims(checkable_data, -1)
# print('Checkable Data: ' + str(checkable_data.shape))
# print(checkable_data)
full_image = cv2.imread(args.image)
gray_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
face_only = excise_face(gray_image, CascadeClassifier('haarcascade_frontalface_default.xml'))
print(face_only.shape)
print(face_only)
# face_only -= numpy.mean(face_only,)
face_only -= face_only.mean()
face_only /= face_only.std()
cropped_data = numpy.expand_dims(face_only, -1)
cropped_data = numpy.expand_dims(cropped_data, 0)
print(cropped_data.shape)
print(cropped_data)
# normalize(cropped_data)
checkable_data = cropped_data.copy()
# normalize(cropped_data, checkable_data, alpha=-128, beta=127, dtype=cv2.CV_32F)
print(checkable_data.shape)
print(checkable_data)

# Prepare Model
model = acquire_model(args.timestamp)

#print(checkable_data.shape)
#model.summary()

# Standardise
# checkable_data -= numpy.mean(checkable_data, 0)
# checkable_data /= numpy.std(checkable_data, 0)

# Make Prediction
prediction = model.predict(checkable_data).tolist()[0]
all_predictions = model.predict(checkable_data)
print('Predictions: ' + str(all_predictions))
# print(str(model.predict(checkable_data.tolist())))
# print(prediction)
#print(prediction[0])
#print(prediction[2])
#print(prediction[6])

# Show Image and Confidence Plot
figure, (axis_image, axis_confidence) = pyplot.subplots(1,2)
axis_image.imshow(checkable_data[0].reshape(FACE_SIZE), interpolation='none', cmap='gray')
axis_confidence.barh(numpy.arange(len(prediction)), width=prediction)
axis_confidence.yaxis.set_major_locator(MaxNLocator(integer=True))
# axis_confidence.set_yticklabels(['','neutral','surprise','sadness','fear','anger','disgust','joy'])
axis_confidence.set_xlim([0,1])
axis_confidence.xaxis.set_major_locator(MaxNLocator(10))
axis_confidence.grid(True, which='major', axis='x', linestyle='--')
pyplot.subplots_adjust(wspace=0.5)
pyplot.show()