import logging

import tensorflow as tf
from os.path import exists
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, ReLU, Dropout, ActivityRegularization, LayerNormalization, LeakyReLU
import pickle


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def model_factory():
    model = Sequential()
    model.add(Conv1D(64, 2, activation="relu", input_shape=(901, 1)))
    model.add(LeakyReLU(50))
    model.add(Dense(30))
    model.add(Dropout(.05))
    model.add(MaxPooling1D())
    model.add(Dense(30))
    model.add(Dropout(.025))
    model.add(Flatten())
    model.add(LayerNormalization())
    model.add(ActivityRegularization(l2=1))
    model.add(Dense(5, activation="linear"))
    model.add(LayerNormalization())
    model.add(Dense(1, activation="linear"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    return model


# Load in data
base_bath = "Data/pickles/"  # change as needed!

train_x = load(base_bath + "train_x.pck")
train_y = load(base_bath + "train_y.pck")

test_x = load(base_bath + "test_x.pck")
test_y = load(base_bath + "test_y.pck")

validate_x = load(base_bath + "validate_x.pck")
validate_y = load(base_bath + "validate_y.pck")

# Uncomment out to discover shape of the data files
# print(len(validate_x))
# print(len(test_x))
# print(len(train_x))
# print(train_x.shape)

# Reshape datat so keras can use it
train_x = train_x.reshape([-1, 901, 1])
test_x = test_x.reshape([-1, 901, 1])
validate_x = validate_x.reshape([-1, 901, 1])

#### CHANGE THIS TO WORK WITH DIFERNET MODELS
model_name = "model4"
####
# check to see if model exists
if exists("keras/" + model_name + "/saved_model.pb"):  # model already exists
    model = tf.keras.models.load_model("keras/" + model_name)
else:  # the model does not exist so we need to make it before we can train
    model = model_factory()

# runs until I halt the program
my_file = open(model_name + ".txt", "w")
while True:
    try:
        model.fit(train_x, train_y, batch_size=16, epochs=10)
        acc = model.evaluate(test_x, test_y)
        print("Loss", acc[0], " Acc: ", acc[1])

        model.save("keras/" + model_name)

        # Discover stats on the model
        game_outcomes = []
        predicted_avg = 0
        pred = model.predict(validate_x)
        validation = 0

        for prediction in pred:
            # print(abs(prediction) > .5, prediction)
            if abs(prediction) > .5:
                game_outcomes.append(1)
            else:
                game_outcomes.append(0)

            predicted_avg += abs(prediction)

        for i in range(len(game_outcomes)):
            # print("predicted X: ", game_outcomes[i], "Actual Y:", validate_y[i])
            if game_outcomes[i] == validate_y[i]:
                validation += 1

        print("Validation = ", validation / len(game_outcomes))
        print("Average % for this round = ", predicted_avg / len(game_outcomes))

        my_file.write("Validation = " + str(validation / len(game_outcomes)) + "\tAverage % = " + str(predicted_avg / len(game_outcomes))
                      + "\tloss = " + str(acc[0]) + "\n")
        my_file.flush()
    except KeyboardInterrupt:
        break

my_file.close()