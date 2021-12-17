import json
import pickle

import pandas as pd
import numpy as np


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

# attempt to standardize
# x = (x - x.min()) / (x.max() - x.min())

del X_2015_2016, X_2016_2017, X_2017_2018, X_2018_2019
del Y_2015_2016, Y_2016_2017, Y_2017_2018, Y_2018_2019

print("Shape of X: ", x.shape)
print("Shape of Y: ", y.shape)
x = x.replace(np.nan, 0)
x = x.drop(columns=[15])
train_x = pd.DataFrame()
test_x = pd.DataFrame()
validate_x = pd.DataFrame()

train_y = pd.DataFrame()
test_y = pd.DataFrame()
validate_y = pd.DataFrame()

# this for loop takes about 2 minutes to run...
print("entering loop that takes forever")
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

train_x = train_x.to_numpy(dtype=float)
train_y = train_y.to_numpy(dtype=float)

test_x = test_x.to_numpy(dtype=float)
test_y = test_y.to_numpy(dtype=float)

validate_x = validate_x.to_numpy(dtype=float)
validate_y = validate_y.to_numpy(dtype=float)

# Make pickle files so we only have to do all of this once
###########################################
# training data
filename = 'train_x.pck'
outfile = open("Data/pickles/" + filename, 'wb')
pickle.dump(train_x, outfile)
outfile.close()
filename = 'train_y.pck'
outfile = open("Data/pickles/" + filename, 'wb')
pickle.dump(train_y, outfile)
outfile.close()

# test data
filename = 'test_x.pck'
outfile = open("Data/pickles/" + filename, 'wb')
pickle.dump(test_x, outfile)
outfile.close()
filename = 'test_y.pck'
outfile = open("Data/pickles/" + filename, 'wb')
pickle.dump(test_y, outfile)
outfile.close()

# validation
filename = 'validate_x.pck'
outfile = open("Data/pickles/" + filename, 'wb')
pickle.dump(validate_x, outfile)
outfile.close()
filename = 'validate_y.pck'
outfile = open("Data/pickles/" + filename, 'wb')
pickle.dump(validate_y, outfile)
outfile.close()

#
# train_x = train_x.reshape([-1, 902, 1])
# test_x = test_x.reshape([-1, 902, 1])