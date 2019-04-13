"""

Please see selected notes on architecture for a deeper explanation on pre-processing choices. 

To avoid requiring a configuration file with environmental variable paths, we assume you are running this script from its parent directory. 

The path structure for images is described below, though they are empty for the purposes of this repository. 

TRAIN_PATH: path to the folder containing the training images, corresponding to the train folder
TEST_PATH: path to the folder containing the test images
LABEL_PATH: path to the json file contains the correct keypoints for images
MODEL_PATH: path to the keras h5 file contains the trained model used for inference

Make sure you have all packages installed in your virtual env: numpy, pandas, keras, pillow, matplotlib

To make prediction on new images:

1. Drop them to the TEST_PATH
2. Modify the LABEL_PATH json file if you have the correct keypoints
3. Run this script
"""

import json
import os
import glob
import pandas as pd
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

TRAIN_PATH = os.getcwd()+"/train/"
TEST_PATH = os.getcwd()+"/test/"
LABEL_PATH = os.getcwd()+"/keypoint_labels.json"
MODEL_PATH = os.getcwd()+"/model/keypoint.h5"

SAMPLE_SIZE = 150
HEIGHT = 688
WIDTH = 1032
LABELS = [
        "bl_upper_block_x",
        "bl_upper_block_y",
        "ul_bottom_block_x",
        "ul_bottom_block_y",
        "ur_bottom_block_x",
        "ur_bottom_block_y"
    ]

LABEL_MAP = {
    "bl_upper_block_x": ["bl_upper_block", "x"],
    "bl_upper_block_y": ["bl_upper_block", "y"],
    "ul_bottom_block_x": ["ul_bottom_block", "x"],
    "ul_bottom_block_y": ["ul_bottom_block", "y"],
    "ur_bottom_block_x": ["ur_bottom_block", "x"],
    "ur_bottom_block_y": ["ur_bottom_block", "y"]
}

with open(LABEL_PATH) as f:
    LABEL_DICT = json.load(f)


def load_data(img_lst):
    """
    Load img_name data from TEST_PATH into numpy arrays

    X: 4-d numpy array (1, Nrow, Ncol, 1)
    y: 2-d numpy array (1, Nkeypoints*2)

    return X, y
    """

    X = np.stack([img_to_array(load_img(TEST_PATH + img_name, color_mode = "grayscale")).reshape(HEIGHT, WIDTH, 1) for img_name in img_lst])/255.
    if all([LABEL_DICT.get(img_name) for img_name in img_lst]):
        y = np.vstack([np.array([LABEL_DICT[img_name][v[0]][v[1]] for k, v in LABEL_MAP.items()]).reshape(1, 6) for img_name in img_lst])
        y = y.astype(np.float32)
        y[:, 0::2] = (y[:, 0::2] - WIDTH/2.) / (WIDTH/2.)
        y[:, 1::2] = (y[:, 1::2] - HEIGHT/2.) / (HEIGHT/2.)
    else:
        y = None
    return X, y

def plot_sample_eval(X, y_pred, y_val, axs):
    """
    y is rescaled to range between -1 and 1
    """

    axs.imshow(X.reshape(HEIGHT, WIDTH), cmap="gray")
    axs.scatter(WIDTH/2*y_pred[0::2]+ WIDTH/2, HEIGHT/2*y_pred[1::2]+ HEIGHT/2, c="r", marker="o")
    if y_val is not None:
        axs.scatter(WIDTH/2*y_val[0::2]+ WIDTH/2, HEIGHT/2*y_val[1::2]+ HEIGHT/2, c="b", marker="o")


def keypoint_serve(model):
    """
    This function load all images in TEST_PATH, predict and plot the first image

    return: loss (mean_squared_error) for y (in [-1, 1] scale)
    """
    if os.path.exists(os.getcwd()+"/test/.DS_Store"):
        os.remove(os.getcwd()+"/test/.DS_Store")

    X_val, y_val = load_data([x for x in os.listdir(TEST_PATH)])
    y_pred = model.predict(X_val)
    val_loss = np.mean((y_pred - y_val)**2)
    if y_val is not None:
        fig = plt.figure(figsize=(7, 7))
        fig.subplots_adjust(hspace=0.13, wspace=0.0001,
                            left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        plot_sample_eval(X_val[0], y_pred[0], y_val[0], ax)
        ax.set_title("Sample picture")
        plt.show()
    return val_loss

model = load_model(MODEL_PATH)
val_loss = keypoint_serve(model)
print("Loss value for y in scale [-1, 1] is: %s" %(val_loss))