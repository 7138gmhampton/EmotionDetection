"""Contains constants and hyperparameters for the training of the Facial \
    Expression Recognition classifier"""
MODEL_DIRECTORY = 'models'
FACE_EXTRACTOR = 'haarcascade_frontalface_default.xml'
IMAGE_DIRECTORIES = [('000 neutral', 0),
                     ('001 surprise', 1),
                     ('002 sadness', 2),
                     ('003 fear', 3),
                     ('004 anger', 4),
                     ('005 disgust', 5),
                     ('006 joy', 6)]

# IMAGE_HEIGHT = 490
# IMAGE_WIDTH = 640
# Can be scaled down by 2, 5 or 10
# SCALE_FACTOR = 5

# ROWS = int(IMAGE_HEIGHT/SCALE_FACTOR)
# COLS = int(IMAGE_WIDTH/SCALE_FACTOR)

# Data Augmentation
CLOCKWISE = 5
ANTICLOCKWISE = -CLOCKWISE

# Face Only Bounds
FACE_BOUND = 300
SCALE_DOWN_FACTOR = 3
FACE_BOUND_SCALED = int(FACE_BOUND/SCALE_DOWN_FACTOR)
FACE_SIZE = (FACE_BOUND_SCALED, FACE_BOUND_SCALED)

# Model Hyperparameters
NO_OF_FEATURES = 64
NO_OF_LABELS = 7
BATCH_SIZE = 16
NO_OF_EPOCHS = 1
DROPOUT = 0.05
