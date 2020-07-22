import numpy, cv2

from mtcnn.mtcnn import MTCNN
from keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot

print(cv2.__version__)

# Load an Image
raw_image = load_img('example.png', color_mode='grayscale')
raw_image = img_to_array(raw_image).squeeze()
raw_image = raw_image.astype(numpy.uint8)
print(raw_image.shape)
print(raw_image)
# other_image = pyplot.imread('example.png')
# print(other_image.shape)
other_image = cv2.imread('example.png')
gray_image = cv2.cvtColor(other_image, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)
print(gray_image)

# Extract Face
# detector = MTCNN()
# results = detector.detect_faces(other_image)
# print(results)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(raw_image, 1.3, 1)
# print(faces[0])
# print(len(faces))

# Output Result as Array

# Display Result
cv2.imshow('output', other_image)
cv2.waitKey()
cv2.destroyAllWindows()