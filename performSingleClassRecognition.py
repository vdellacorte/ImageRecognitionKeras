from keras.applications import vgg16
import numpy as np
from transferLearner import load_trained_model
from keras.preprocessing import image
import config

model = load_trained_model()

# target_size is the image size
img = image.load_img(config.TEST_IMAGE, target_size=config.IMAGE_DIMENSION)
image_array = image.img_to_array(img)
images = np.expand_dims(image_array, axis=0)
images = vgg16.preprocess_input(images)

input_shape = config.IMAGE_DIMENSION + (3,)
feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
features = feature_extraction_model.predict(images)

results = model.predict(features)

print("Likelihood to be of the right class: {}%".format(int(results[0][0] * 100)))
