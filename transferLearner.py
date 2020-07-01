from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.models import model_from_json

def load_trained_model():
    f = Path("model_structure.json")
    model_structure = f.read_text()
    model = model_from_json(model_structure)
    model.load_weights("model_weights.h5")
    return model


class TransferLearner:

    def __init__(self, number_of_categories, input_shape, neurons):
        self.__model = self.__build_classifier_layer(number_of_categories, input_shape, neurons)

    def train_classifier_layer(self, x_train, y_train):
        self.__model.fit(
            x_train,
            y_train,
            epochs=10,
            shuffle=True
        )
        self.__save_trained_model(self.__model)
        return self.__model

    def __build_classifier_layer(self, number_of_categories, input_shape, neurons):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(number_of_categories, activation='sigmoid'))
        if number_of_categories == 2:
            loss = "binary_crossentropy"
        else:
            loss = 'categorical_crossentropy'
        model.compile(
            loss=loss,
            optimizer="adam",
            metrics=['accuracy']
        )
        return model

    def __save_trained_model(self, model):
        model_structure = model.to_json()
        f = Path("model_structure.json")
        f.write_text(model_structure)
        model.save_weights("model_weights.h5")
