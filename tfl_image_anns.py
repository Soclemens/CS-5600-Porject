import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def make_image_ann_model():
    input_layer = input_data(shape=[None, 902, 1])
    fc_layer_1 = fully_connected(input_layer, 1000,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 1,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.1)
    model = tflearn.DNN(network)
    return model


def load_image_ann_model(model_path):
    input_layer = input_data(shape=[None, 902, 1])
    fc_layer_1 = fully_connected(input_layer, 1000,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 1,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model


# test a tfl network model on valid_X and valid_Y.
def test_tfl_image_ann_model(network_model, valid_X, valid_Y):
    results = []
    for i in range(len(valid_X)):
        prediction = network_model.predict(valid_X[i].reshape([-1, 902, 1]))
        results.append(prediction[0][0] == valid_Y[i][0])
    return float(sum((np.array(results) == True))) / float(len(results))


# train a tfl model on train_X, train_Y, test_X, test_Y.
def train_tfl_image_ann_model(model, train_X, train_Y, test_X, test_Y, num_epochs=2, batch_size=10):
  tf.compat.v1.reset_default_graph()
  model.fit(train_X, train_Y, n_epoch=num_epochs,
            shuffle=True,
            validation_set=(test_X, test_Y),
            show_metric=True,
            batch_size=batch_size,
            run_id='image_ann_model')


# validating is testing on valid_X and valid_Y.
def validate_tfl_image_ann_model(model, valid_X, valid_Y):
    return test_tfl_image_ann_model(model, valid_X, valid_Y)


base_bath = "Data/pickles/"  # change as needed!

train_x = load(base_bath + "train_x.pck")
train_y = load(base_bath + "train_y.pck")

test_x = load(base_bath + "test_x.pck")
test_y = load(base_bath + "test_y.pck")

validate_x = load(base_bath + "validate_x.pck")
validate_y = load(base_bath + "validate_y.pck")

train_x = train_x.reshape([-1, 902, 1])
test_x = test_x.reshape([-1, 902, 1])