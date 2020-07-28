"""Contains functions to build various models for Facial Expression Recognition"""
import os

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
# pylint: disable=wrong-import-position
from keras.models import Sequential, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout,\
     Flatten, Dense, Input, concatenate
from keras.regularizers import l2
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from hyper import NO_OF_FEATURES, FACE_BOUND_SCALED, DROPOUT, NO_OF_LABELS

def build_shanks():
    """Builds model based on one from Nishank Sharma - \
        github.com/gitshanks/fer2013"""
    model = Sequential()

    # First Feature Extraction
    model.add(Conv2D(NO_OF_FEATURES, kernel_size=(3, 3), activation='relu',
                     input_shape=(FACE_BOUND_SCALED, FACE_BOUND_SCALED, 1),
                     data_format='channels_last', kernel_regularizer=l2()))
    model.add(Conv2D(NO_OF_FEATURES, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(DROPOUT))

    # Second Feature Extraction
    model.add(Conv2D(2*NO_OF_FEATURES, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*NO_OF_FEATURES, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(DROPOUT))

    # Third Feature Extraction
    model.add(Conv2D(2*2*NO_OF_FEATURES, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*NO_OF_FEATURES, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(DROPOUT))

    # Fourth Feature Extraction
    model.add(Conv2D(2*2*2*NO_OF_FEATURES, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*2*NO_OF_FEATURES, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
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
    # model.compile(loss=categorical_crossentropy, optimizer=Adam(),
    #               metrics=['accuracy'])

    return model

def build_dexpression():
    """Builds DeXpression model - Buckert et al, 2016"""
    image_input = Input(shape=(FACE_BOUND_SCALED, FACE_BOUND_SCALED, 1))
    
    # Model Start
    convolution_1 = Conv2D(NO_OF_FEATURES, (5, 5), padding='valid', 
                           activation='relu', name='convolution-1')(image_input)
    pooling_1 = MaxPooling2D((2, 2), strides=(2, 2), name='pooling-1')(convolution_1)
    start = BatchNormalization()(pooling_1)
    
    # Feature Extraction 1
    convolution_2a = Conv2D(1.5*NO_OF_FEATURES, (1, 1), strides=(1, 1), activation='relu', padding='valid', name='convolution-2a')(start)
    convolution_2b = Conv2D(3.25*NO_OF_FEATURES, (3, 3),strides=(1, 1), activation='relu', padding='valid', name='convolution-2b')(convolution_2a)
    pooling_2a = MaxPooling2D((3, 3), strides=(1, 1), padding='valid', name='pooling-2a')(start)
    convolution_2c = Conv2D(NO_OF_FEATURES, (1 ,1), strides=(1,1), name='convolution-2c')(pooling_2a)
    concatenate_2 = concatenate(inputs=[convolution_2b, convolution_2c], axis=3, name='concatenate-2')
    pooling_2b = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling-2b')(concatenate_2)
    
    # Feature Extraction 2
    convolution_3a = Conv2D(1.5*NO_OF_FEATURES, (1, 1), strides=(1, 1), activation='relu', padding='valid', name='convolution-3a')(pooling_2b)
    convolution_3b = Conv2D(3.25*NO_OF_FEATURES, (3, 3),strides=(1, 1), activation='relu', padding='valid', name='convolution-3b')(convolution_3a)
    pooling_3a = MaxPooling2D((3,3), strides=(1,1), padding='valid', name='pooling-3a')(pooling_2b)
    convolution_3c = Conv2D(NO_OF_FEATURES, (1, 1), strides=(1, 1), name='convolution-3c')(pooling_3a)
    concatenate_3 = concatenate(inputs=[convolution_3b, convolution_3c], axis=3, name='concatenate-3')
    pooling_3b = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling-3b')(concatenate_3)
    
    # Classification
    flatten = Flatten()(pooling_3b)
    classifier = Dense(NO_OF_LABELS, activation='softmax', name='predictions')(flatten)
    
    # Combine Model
    dexpression_model = Model(image_input, classifier, name='deXpression')
    
    return dexpression_model

def prepare_model(selected_model):
    """Build and compile selected model"""
    build_functions = {'shanks':build_shanks, 'dexpression':build_dexpression}
    function_to_use = build_functions.get(selected_model, build_shanks)
    
    model = function_to_use()
    
    # Compile Model
    model.compile(loss=categorical_crossentropy, optimizer=Adam(),
                  metrics=['accuracy'])
    return model