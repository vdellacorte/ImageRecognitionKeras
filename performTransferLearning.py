from transferLearner import TransferLearner
from trainingDataLoader import load_features_target_from_file
import config

x_train, y_train = load_features_target_from_file()

transferLearner = TransferLearner(number_of_categories=config.NUMBER_CLASSES,
                                  input_shape=x_train.shape[1:],
                                  neurons=config.SOFTMAX_HIDDEN_NEURONS)
transferLearner.train_classifier_layer(x_train, y_train)
