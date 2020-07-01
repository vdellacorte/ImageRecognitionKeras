from pathlib import Path

# for both training and testing
# it must be at least 48x48
IMAGE_DIMENSION = (64, 64)
IMAGE_FORMAT = ".png"

NUMBER_CLASSES = 1
TRAINING_CLASSES_FOLDERS_PATH = [Path("training_data") / "first_category",
                                 Path("training_data") / "second_category"]
SOFTMAX_HIDDEN_NEURONS = 256

TEST_IMAGE = "test_data/bay.jpg"
