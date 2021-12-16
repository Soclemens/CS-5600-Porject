import numpy as np
import tensorflow as tf
import pandas as pd
import json
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression


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


def flatten_data(data_to_flatten):
    data = pd.DataFrame.from_dict(data_to_flatten, orient='index')
    return data


# load data from json files
year_2015_2016_X = json.load(open('Data/flattened_data/2015-2016_X.json'))
year_2016_2017_X = json.load(open('Data/flattened_data/2016-2017_X.json'))
year_2017_2018_X = json.load(open('Data/flattened_data/2017-2018_X.json'))
year_2018_2019_X = json.load(open('Data/flattened_data/2018-2019_X.json'))

# load data for the y from json files
year_2015_2016_Y = json.load(open('Data/flattened_data/2015-2016_Y.json'))
year_2016_2017_Y = json.load(open('Data/flattened_data/2016-2017_Y.json'))
year_2017_2018_Y = json.load(open('Data/flattened_data/2017-2018_Y.json'))
year_2018_2019_Y = json.load(open('Data/flattened_data/2018-2019_Y.json'))

# Flatten data to where each vector is a row IE One game is one row
X_2015_2016 = flatten_data(year_2015_2016_X)
X_2016_2017 = flatten_data(year_2016_2017_X)
X_2017_2018 = flatten_data(year_2017_2018_X)
X_2018_2019 = flatten_data(year_2018_2019_X)


Y_2015_2016 = flatten_data(year_2015_2016_Y)
Y_2016_2017 = flatten_data(year_2016_2017_Y)
Y_2017_2018 = flatten_data(year_2017_2018_Y)
Y_2018_2019 = flatten_data(year_2018_2019_Y)

# Clear used data that is no longer useful
del year_2015_2016_X
del year_2016_2017_X
del year_2017_2018_X
del year_2015_2016_Y
del year_2016_2017_Y
del year_2017_2018_Y

# Just the columns need to be the same size as there can be different games played year to year
# the rows are a game so there columns need to be the same size
# print(X_2015_2016.shape)
# print(X_2016_2017.shape)
# print(X_2017_2018.shape)
# print(X_2018_2019.shape)


x = pd.concat([X_2015_2016, X_2016_2017, X_2017_2018, X_2018_2019])
y = pd.concat([Y_2015_2016, Y_2016_2017, Y_2017_2018, Y_2018_2019])

del X_2015_2016, X_2016_2017, X_2017_2018, X_2018_2019
del Y_2015_2016, Y_2016_2017, Y_2017_2018, Y_2018_2019

print("Shape of X: ", x.shape)
print("Shape of Y: ", y.shape)
x = x.replace(np.nan, 0)

train_x = pd.DataFrame()
test_x = pd.DataFrame()
validate_x = pd.DataFrame()

train_y = pd.DataFrame()
test_y = pd.DataFrame()
validate_y = pd.DataFrame()

# this for loop takes about 2 minutes to run...
for i in range(x.shape[0]):
    if i % 10 >= 0 and i % 10 <= 7:

        train_x = train_x.append(x.iloc[i].transpose())  # transposing here to keep each game as a row
        train_y = train_y.append(y.iloc[i])
    elif i % 10 == 8:
        test_x = test_x.append(x.iloc[i].transpose())  # transposing here to keep each game as a row
        test_y = test_y.append(y.iloc[i])
    elif i % 10 == 9:
        validate_x = validate_x.append(x.iloc[i].transpose())  # transposing here to keep each game as a row
        validate_y = validate_y.append(y.iloc[i])

print(validate_x.shape)
print(train_x.shape)
print(test_x.shape)

print(validate_y.shape)
print(train_y.shape)
print(test_y.shape)

train_x = train_x.to_numpy()
train_y = train_y.to_numpy()

test_x = test_x.to_numpy()
test_y = test_y.to_numpy()

validate_x = validate_x.to_numpy()
validate_y = validate_y.to_numpy()


train_x = train_x.reshape([-1, 902, 1])
test_x = test_x.reshape([-1, 902, 1])

# validate_x = validate_x.transpose()
# validate_y = validate_y.transpose()