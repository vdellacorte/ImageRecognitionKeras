from keras.applications import vgg16
import config
from trainingDataLoader import TrainingDataLoader, save_features_target_to_file

isSingleClass = True if config.NUMBER_CLASSES == 1 else False
trainingDataLoader = TrainingDataLoader(is_single_class=isSingleClass, image_format=config.IMAGE_FORMAT)
x_train, y_train = trainingDataLoader.load_data(config.TRAINING_CLASSES_FOLDERS_PATH)
x_train = vgg16.preprocess_input(x_train)

input_shape = config.IMAGE_DIMENSION + (3,)
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
features_x = pretrained_nn.predict(x_train)

save_features_target_to_file(features_x, y_train)
