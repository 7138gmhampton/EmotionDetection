import argparse, os, hyper

from keras.models import model_from_json, Sequential

def acquire_model(timestamp):
    directory = hyper.MODEL_DIRECTORY
    model_name = timestamp + '_model.json'
    weights_name = timestamp + '_weights.h5'
    
    with open(os.path.join(directory, model_name), 'r') as json_model:
        #return model_from_json(json_model.read())
        model = model_from_json(json_model.read())
    model.load_weights(os.path.join(directory, weights_name))

    return model

# Set up Command Line Arguments
parser = argparse.ArgumentParser(description='Apply test examples to trained model.')
parser.add_argument('-t', '--timestamp', help='The datetime for the model', required=True)
args = parser.parse_args()

#print(args.model)

# Prepare Model
timestamp = args.timestamp

model = acquire_model(timestamp)
print(' -- Model loaded from file --')
