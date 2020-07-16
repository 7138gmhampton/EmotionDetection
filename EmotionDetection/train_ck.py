import numpy, hyper, os
import matplotlib.pyplot as pyplot

from sklearn.model_selection import train_test_split

# Change Keras Backend
#os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from kerastuner.engine.hyperparameters import HyperParameter

# Hyperparameters
no_of_features = hyper.NO_OF_FEATURES
no_of_labels = hyper.NO_OF_LABELS
batch_size = hyper.BATCH_SIZE
no_of_epochs = hyper.NO_OF_EPOCHS
rows, cols = hyper.ROWS, hyper.COLS

def build_model():
    model = Sequential()

    #features_from_filters = hp.Int('filters', min_value=32, max_value=512, step=32)
    features_from_filters = no_of_features
    
    model.add(Conv2D(features_from_filters, kernel_size=(3, 3), activation='relu', 
        input_shape=(rows, cols, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*2*features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*2*2*features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*2*features_from_filters, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(2*2*2*features_from_filters, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2*2*features_from_filters, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2*features_from_filters, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(no_of_labels, activation='softmax'))

    #model.summary()

    # Compile Model
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

    return model

# Load Training Data
data = numpy.load('ck_data.npy')
labels = numpy.load('ck_labels.npy')

# Check Load from File
#for iii in range(3):
#    pyplot.figure(iii).suptitle(labels[iii])
#    pyplot.imshow(data[iii].reshape((HEIGHT,WIDTH)), interpolation='none', cmap='gray')
#pyplot.show()

# Standardise
data -= numpy.mean(data, 0)
data /= numpy.std(data, 0)

# Check Standardise
#for iii in range(3):
#    pyplot.figure(iii).suptitle(labels[iii])
#    pyplot.imshow(data[iii].reshape((HEIGHT,WIDTH)), interpolation='none', cmap='gray')
#pyplot.show()

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

# CNN Pipeline

#model = Sequential()

#model.add(Conv2D(no_of_features, kernel_size=(3, 3), activation='relu', 
#    input_shape=(rows, cols, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
#model.add(Conv2D(no_of_features, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Dropout(0.5))

#model.add(Conv2D(2*no_of_features, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(2*no_of_features, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Dropout(0.5))

#model.add(Conv2D(2*2*no_of_features, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(2*2*no_of_features, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Dropout(0.5))

#model.add(Conv2D(2*2*2*no_of_features, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(2*2*2*no_of_features, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Dropout(0.5))

#model.add(Flatten())

#model.add(Dense(2*2*2*no_of_features, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(2*2*no_of_features, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(2*no_of_features, activation='relu'))
#model.add(Dropout(0.5))

#model.add(Dense(no_of_labels, activation='softmax'))

#model.summary()

#model = build_model()

# Train Model
#model.fit(numpy.array(data_train), 
#          numpy.array(labels_train), 
#          batch_size=batch_size, 
#          epochs=no_of_epochs,
#          verbose=1,
#          validation_data=(numpy.array(data_valid), numpy.array(labels_valid)),
#          shuffle=True)

#from kerastuner.tuners import RandomSearch

#tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=1, executions_per_trial=1,
#                     directory='tuning_model')

#tuner.search_space_summary()

#tuner.search(numpy.array(data_train),
#             numpy.array(labels_train),
#             batch_size=batch_size,
#             epochs=no_of_epochs,
#             verbose=1,
#             validation_data=(numpy.array(data_valid), numpy.array(labels_valid)))

model = build_model()

model.fit(numpy.array(data_train), 
          numpy.array(labels_train), 
          batch_size=batch_size, 
          epochs=no_of_epochs,
          verbose=1,
          validation_data=(numpy.array(data_valid), numpy.array(labels_valid)),
          shuffle=True)