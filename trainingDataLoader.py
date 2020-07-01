import numpy as np
from keras.preprocessing import image
import joblib


def load_features_target_from_file():
    x_train = joblib.load("x_train.dat")
    y_train = joblib.load("y_train.dat")
    return x_train, y_train


def save_features_target_to_file(features_x, y_train):
    joblib.dump(features_x, "x_train.dat")
    joblib.dump(y_train, "y_train.dat")


class TrainingDataLoader:

    def __init__(self, is_single_class, image_format):
        self.__is_single_class = is_single_class
        self.__image_format = image_format

    def load_data(self, category_paths):
        if self.__is_single_class:
            return self.__load_for_binary_classes(category_paths)
        else:
            return self.__load_for_categorical_classes(category_paths)

    def __load_for_binary_classes(self, category_paths):
        images = []
        labels = []
        for i in range(len(category_paths)):
            for img in category_paths[i].glob("*" + self.__image_format):
                img = image.load_img(img)
                image_array = image.img_to_array(img)
                images.append(image_array)
                labels.append(i)

        x_train = np.array(images)
        y_train = np.array(labels)
        return x_train, y_train

    def __load_for_categorical_classes(self, category_paths):
        return []
