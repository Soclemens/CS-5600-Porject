import tflearn
import tensorflow as tf
from tfl_image_anns import *
from os.path import exists
from tflearn.layers.core import input_data, fully_connected, highway, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d, atrous_conv_2d, upscore_layer
from tflearn.layers import normalization
from tflearn.layers.estimator import regression
# Models

#####################################################
def model_1():
    net = input_data(shape=[None, 901, 1])
    net = fully_connected(net, 1800)
    net = fully_connected(net, 1000)
    net = fully_connected(net, 100)
    net = fully_connected(net, 1, activation="softmax")
    network = regression(net, optimizer='sgd', loss='binary_crossentropy', learning_rate=0.01)

    return network


def model_2():
    net = input_data(shape=[None, 901, 1])
    net = fully_connected(net, 1800)
    net = fully_connected(net, 2500)
    net = dropout(net, .25)
    net = fully_connected(net, 100)
    net = fully_connected(net, 1, activation="softmax")
    network = regression(net, optimizer='sgd', loss='binary_crossentropy', learning_rate=0.01)

    return network
#####################################################

# trainers
#####################################################
def train_models(pair):
    if exists("models/" + pair[1] + "/model.tfl.meta"):  # model already exists
        img_ann = load_image_ann_model("models/" + pair[1] + "/model.tfl", pair[0]())
    else:  # the model does not exist so we need to make it before we can train
        img_ann = make_image_ann_model(pair[0]())

    # train
    train_tfl_image_ann_model(img_ann, train_x, train_y, test_x, test_y)

    # save
    img_ann.save("models/" + pair[1] + "/model.tfl")

    return validate_tfl_image_ann_model(img_ann, validate_x, validate_y)
#####################################################


def train_all_models():
    known_training = [(model_1, "model_1"), (model_2, "model_2")]  # (model_1, "model_1"),
    validations = []
    for pair in known_training:
        validations.append(train_models(pair))



    return validations

def rank_models():
    known_models = [("models/model_2/model.tfl", model_2)]
    for pair in known_models:
        img_ann = load_image_ann_model(pair[0], pair[1]())
        print(pair[0].split("/")[1] + ": ", validate_tfl_image_ann_model(img_ann, validate_x, validate_y))


print(train_all_models())
