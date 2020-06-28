from trainingDataLoader import TrainingDataLoader, save_features_target_to_file
from pathlib import Path
from keras.applications import vgg16

class_zero_path = Path("training_data") / "first_category"
class_one_path = Path("training_data") / "second_category"

trainingDataLoader = TrainingDataLoader(is_binary=True)
x_train, y_train = trainingDataLoader.load_data([class_zero_path, class_one_path])
x_train = vgg16.preprocess_input(x_train)

pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
features_x = pretrained_nn.predict(x_train)

save_features_target_to_file(features_x, y_train)
