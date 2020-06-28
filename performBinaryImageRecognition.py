from keras.applications import vgg16
import numpy as np
from transferLearner import load_trained_model
from keras.preprocessing import image

model = load_trained_model()

img = image.load_img("test_data/dog.png", target_size=(64, 64))
image_array = image.img_to_array(img)
images = np.expand_dims(image_array, axis=0)
images = vgg16.preprocess_input(images)

feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
features = feature_extraction_model.predict(images)

results = model.predict(features)
single_result = results[0][0]

print("Likelihood to be a dog: {}%".format(int(single_result * 100)))
