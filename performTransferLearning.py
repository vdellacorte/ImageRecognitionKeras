from transferLearner import TransferLearner
from trainingDataLoader import load_features_target_from_file

x_train, y_train = load_features_target_from_file()

transferLearner = TransferLearner(number_of_categories=2, input_shape=x_train.shape[1:])
transferLearner.train_classifier_layer(x_train, y_train)
