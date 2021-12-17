import numpy as np
import tensorflow as tf
import pickle
import tflearn


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def make_image_ann_model(model):
    network = model
    model = tflearn.DNN(network)
    return model


def load_image_ann_model(model_path, network):
    model = tflearn.DNN(network)
    model.load(model_path)
    return model


# test a tfl network model on valid_X and valid_Y.
def test_tfl_image_ann_model(network_model, valid_X, valid_Y):
    results = []
    for i in range(len(valid_X)):
        prediction = network_model.predict(valid_X[i].reshape([-1, 901, 1]))
        print(prediction[0][0])
        results.append(prediction[0][0] == valid_Y[i][0])
    return float(sum((np.array(results) == True))) / float(len(results))


# train a tfl model on train_X, train_Y, test_X, test_Y.
def train_tfl_image_ann_model(model, train_X, train_Y, test_X, test_Y, num_epochs=1, batch_size=10):
  # tf.compat.v1.reset_default_graph()
  model.fit(train_X, train_Y, n_epoch=num_epochs,
            validation_set=(test_X, test_Y),
            show_metric=True,
            batch_size=batch_size,
            run_id='image_ann_model')

# validating is testing on valid_X and valid_Y.
def validate_tfl_image_ann_model(model, valid_X, valid_Y):
    return test_tfl_image_ann_model(model, valid_X, valid_Y)


tflearn.init_graph(num_cores=8, gpu_memory_fraction=.25)  # guess this allows for faster training

base_bath = "Data/pickles/"  # change as needed!

train_x = load(base_bath + "train_x.pck")
train_y = load(base_bath + "train_y.pck")

test_x = load(base_bath + "test_x.pck")
test_y = load(base_bath + "test_y.pck")

validate_x = load(base_bath + "validate_x.pck")
validate_y = load(base_bath + "validate_y.pck")

print(len(validate_x))
print(len(test_x))
print(len(train_x))

train_x = train_x.reshape([-1, 901, 1])
test_x = test_x.reshape([-1, 901, 1])