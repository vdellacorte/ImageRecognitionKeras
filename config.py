from pathlib import Path

# for both training and testing
# it must be at least 48x48
IMAGE_DIMENSION = (512, 512)
IMAGE_FORMAT = ".jpg"

NUMBER_CLASSES = 1
TRAINING_CLASSES_FOLDERS_PATH = [Path("training_data") / "first_class",
                                 Path("training_data") / "second_class"]
# for transfer learning
SOFTMAX_HIDDEN_NEURONS = 256

TEST_IMAGE = "test_data/IMG_20190530_200448.jpg"
