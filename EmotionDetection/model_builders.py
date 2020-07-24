import os
from hyper import NO_OF_FEATURES, FACE_BOUND_SCALED, DROPOUT, NO_OF_LABELS

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.regularizers import l2
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

def build_shanks():
    model = Sequential()

    # First Feature Extraction
    model.add(Conv2D(NO_OF_FEATURES, kernel_size=(3,3), activation='relu',
        input_shape=(FACE_BOUND_SCALED, FACE_BOUND_SCALED,1),
        data_format='channels_last', kernel_regularizer=l2()))
    model.add(Conv2D(NO_OF_FEATURES,kernel_size=(3,3),activation='relu',
        padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(DROPOUT))

    # Second Feature Extraction
    model.add(Conv2D(2*NO_OF_FEATURES,kernel_size=(3,3),activation='relu',
        padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*NO_OF_FEATURES,kernel_size=(3,3),activation='relu',
        padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(DROPOUT))

    # Third Feature Extraction
    model.add(Conv2D(2*2*NO_OF_FEATURES,kernel_size=(3,3),activation='relu',
        padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*NO_OF_FEATURES,kernel_size=(3,3),activation='relu', 
        padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(DROPOUT))

    # Fourth Feature Extraction
    model.add(Conv2D(2*2*2*NO_OF_FEATURES,kernel_size=(3,3),activation='relu', 
        padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*2*NO_OF_FEATURES,kernel_size=(3,3),activation='relu', 
        padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(DROPOUT))

    # Classifier
    model.add(Flatten())
    model.add(Dense(2*2*2*NO_OF_FEATURES, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(2*2*NO_OF_FEATURES, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(2*NO_OF_FEATURES, activation='relu'))
    model.add(Dropout(DROPOUT))

    # Prediction
    model.add(Dense(NO_OF_LABELS, activation='softmax'))

    # Compile Model
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), 
        metrics=['accuracy'])

    return model