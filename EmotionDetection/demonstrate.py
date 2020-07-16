import argparse, hyper, numpy, os
import matplotlib.pyplot as pyplot

from keras.preprocessing.image import load_img, img_to_array
from keras.models import model_from_json
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
image_size = (hyper.ROWS, hyper.COLS)
image_to_check = load_img(args.image, color_mode='grayscale', target_size=image_size)
image_array = img_to_array(image_to_check)
image_array = numpy.asarray(image_array).squeeze()
data_wrapper = [image_array.astype('float32')]
checkable_data = numpy.asarray(data_wrapper)
checkable_data = numpy.expand_dims(checkable_data, -1)

# Prepare Model
model = acquire_model(args.timestamp)

#print(checkable_data.shape)
#model.summary()

# Make Prediction
prediction = model.predict(checkable_data).tolist()[0]
#print(prediction[0])
#print(prediction[2])
#print(prediction[6])

# Show Image and Confidence Plot
figure, (axis_image, axis_confidence) = pyplot.subplots(1,2)
axis_image.imshow(checkable_data[0].reshape(image_size), interpolation='none', cmap='gray')
axis_confidence.bar(numpy.arange(len(prediction)), height=prediction)
axis_confidence.xaxis.set_major_locator(MaxNLocator(integer=True))
pyplot.show()