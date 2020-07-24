"""Train a model on the preprocessed Cohn-Kanade Dataset and save that model"""
import os
import argparse
# import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
import numpy
import hyper
from hyper import FACE_BOUND_SCALED

# Change Keras Backend
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from model_builders import build_shanks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from kerastuner.engine.hyperparameters import HyperParameter
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras.optimizers
import tensorflow

# Command Line Parameter
parser = argparse.ArgumentParser(description='Train CNN model with Cohn-Kanade dataset.')
parser.add_argument('-e', '--epochs', type=int , required=True)
args = parser.parse_args()

# Hyperparameters
no_of_features = hyper.NO_OF_FEATURES
no_of_labels = hyper.NO_OF_LABELS
batch_size = hyper.BATCH_SIZE
no_of_epochs = args.epochs
rows, cols = hyper.ROWS, hyper.COLS

def build_model():
    model = Sequential()

    features_from_filters = no_of_features
    
    model.add(Conv2D(features_from_filters, kernel_size=(3, 3), activation='relu', 
        input_shape=(FACE_BOUND_SCALED, FACE_BOUND_SCALED, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(hyper.DROPOUT))

    model.add(Conv2D(2*features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(hyper.DROPOUT))

    model.add(Conv2D(2*2*features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(hyper.DROPOUT))

    model.add(Conv2D(2*2*2*features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*2*features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(hyper.DROPOUT))

    model.add(Flatten())

    model.add(Dense(2*2*2*features_from_filters, activation='relu'))
    model.add(Dropout(hyper.DROPOUT))
    model.add(Dense(2*2*features_from_filters, activation='relu'))
    model.add(Dropout(hyper.DROPOUT))
    model.add(Dense(2*features_from_filters, activation='relu'))
    model.add(Dropout(hyper.DROPOUT))

    model.add(Dense(no_of_labels, activation='softmax'))


    # Compile Model
    
    model.compile(loss=categorical_crossentropy, 
    optimizer=Adam(), 
    metrics=['accuracy'])

    return model

def save_trained_model(model, accuracy, validation_accuracy):
    directory = 'models'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    model_name = timestamp + '_model.json'
    weights_name = timestamp + '_weights.h5'
    details_name = timestamp + '_details.txt'

    model_json = model.to_json()
    with open(os.path.join(directory, model_name),'w') as json_file:
        json_file.write(model_json)
    with open(os.path.join(directory, details_name), 'w') as text_file:
        text_file.write('Training Accuracy: ' + '{:1.3f}'.format(accuracy) + '\n')
        text_file.write('Validation Accuracy: ' + '{:1.3f}'.format(validation_accuracy) + '\n')
        text_file.write('Scale Factor: ' + '{:2d}'.format(hyper.SCALE_FACTOR) + '\n')
        text_file.write('Batch Size: ' + '{:3d}'.format(hyper.BATCH_SIZE) + '\n')
        text_file.write('No. of Epochs: ' + '{:3d}'.format(no_of_epochs) + '\n')
        text_file.write('Dropout: ' + '{:1.3f}'.format(hyper.DROPOUT) + '\n')
    model.save_weights(os.path.join(directory, weights_name))

    print(' -- Model Saved --')

# Load Training Data
data = numpy.load('ck_data.npy')
labels = numpy.load('ck_labels.npy')

# Standardise
data -= numpy.mean(data, 0)
data /= numpy.std(data, 0)

# Change Labels to Categorical
from keras.utils import to_categorical
labels = to_categorical(labels)

# Section Data
data_train, data_test, labels_train, labels_test = train_test_split(data,labels, test_size=0.1, random_state=42)
data_train, data_valid, labels_train, labels_valid = train_test_split(data_train,labels_train, test_size=0.1, random_state=42)

# Save Test Data
numpy.save('ck_test_data', data_test)
numpy.save('ck_test_labels', labels_test)
print(' -- Test Data Saved --')

# Train Model
build = build_shanks
model = build()
early_stopper = EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=20, min_delta=0.001)
rate_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
history = model.fit(numpy.array(data_train), 
          numpy.array(labels_train), 
          batch_size=batch_size, 
          epochs=no_of_epochs,
          verbose=1,
          validation_data=(numpy.array(data_valid), numpy.array(labels_valid)),
          shuffle=True,
          callbacks=[rate_reducer])

# Save Model
save_trained_model(model, history.history['acc'][-1], history.history['val_acc'][-1])

# Plot Training History
figure, axis_loss = pyplot.subplots()
axis_loss.plot(history.history['acc'])
pyplot.show()
