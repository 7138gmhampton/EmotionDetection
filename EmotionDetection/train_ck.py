"""Train a model on the preprocessed Cohn-Kanade Dataset and save that model"""
import os
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
# pylint: disable=wrong-import-position
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from metrics import log_details, plot_training
from model_builders import prepare_model
from hyper import BATCH_SIZE, MODEL_DIRECTORY

# Command Line Parameters
parser = argparse.ArgumentParser(description='Train CNN model with Cohn-Kanade dataset.')
parser.add_argument('-e', '--epochs', type=int, required=True)
parser.add_argument('-m', '--model', default='dexpression', required=False)
parser.add_argument('--summary', action='store_true')
args = parser.parse_args()

def provide_data():
    """Load preprocessed data and labels and standardise them for training"""
    data = numpy.load('ck_data.npy')
    labels = numpy.load('ck_labels.npy')

    data -= numpy.mean(data, 0)
    data /= numpy.std(data, 0)

    labels = to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

    numpy.save('ck_test_data', x_test)
    numpy.save('ck_test_labels', y_test)
    print(' -- Test Data Saved --')
    
    return (x_train, y_train, x_valid, y_valid)

def save_trained_model(trained_model, training_history):
    """Save the model and it weights. Also output a text file detailing the \
        hyperparameters and a graphic of the training history"""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    model_name = timestamp + '_model.json'
    weights_name = timestamp + '_weights.h5'
    graph_name = timestamp + '_training.png'

    model_json = trained_model.to_json()
    with open(os.path.join(MODEL_DIRECTORY, model_name), 'w') as json_file:
        json_file.write(model_json)
    log_details(timestamp, training_history, args.epochs, args.model)
    trained_model.save_weights(os.path.join(MODEL_DIRECTORY, weights_name))
    plot_training(training_history, os.path.join(MODEL_DIRECTORY, graph_name))

    print(' -- Model Saved --')

# Load Training Data
# data = numpy.load('ck_data.npy')
# labels = numpy.load('ck_labels.npy')

# # Standardise
# data -= numpy.mean(data, 0)
# data /= numpy.std(data, 0)

# # Change Labels to Categorical
# labels = to_categorical(labels)

# # Section Data
# data_train, data_test, labels_train, labels_test = \
#     train_test_split(data, labels, test_size=0.1, random_state=42)
# data_train, data_valid, labels_train, labels_valid = \
#     train_test_split(data_train, labels_train, test_size=0.1, random_state=42)

# # Save Test Data
# numpy.save('ck_test_data', data_test)
# numpy.save('ck_test_labels', labels_test)
# print(' -- Test Data Saved --')
data_train, labels_train, data_valid, labels_valid = provide_data()

# Train Model
model = prepare_model(args.model, args.summary)
early_stopper = EarlyStopping(monitor='val_loss', mode='max', verbose=1,
                              patience=20, min_delta=0.001)
rate_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
checkpointer = ModelCheckpoint('checkpoint.h5', monitor='val_acc', verbose=1,
                               save_best_only=True, save_weights_only=True)
training = model.fit(numpy.array(data_train),
                     numpy.array(labels_train),
                     batch_size=BATCH_SIZE,
                     epochs=args.epochs,
                     verbose=1,
                     validation_data=(numpy.array(data_valid),
                                      numpy.array(labels_valid)),
                     shuffle=True,
                     callbacks=[rate_reducer, early_stopper, checkpointer])

# Save Model
model.load_weights('checkpoint.h5')
save_trained_model(model, training.history)
