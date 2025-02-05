# import pandas as pd
# import numpy as np

# # load all data
# dataset = pd.read_csv('household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])

# # load and clean-up data
# from numpy import nan
# from numpy import isnan
# from pandas import read_csv

# # fill missing values with a value at the same time one day ago
# def fill_missing(values):
# 	one_day = 60 * 24
# 	for row in range(values.shape[0]):
# 		for col in range(values.shape[1]):
# 			if isnan(values[row, col]):
# 				values[row, col] = values[row - one_day, col]

# # load all data
# dataset = read_csv('household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
# # mark all missing values
# dataset.replace('?', nan, inplace=True)
# # make dataset numeric
# dataset = dataset.astype('float32')
# # fill missing
# fill_missing(dataset.values)
# # add a column for for the remainder of sub metering
# values = dataset.values
# dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])
# # save updated dataset
# dataset.to_csv('household_power_consumption.csv')


# resample minute data to total for each day
from pandas import read_csv
from numpy import split
from numpy import array

# load the new file
dataset = read_csv(
    "household_power_consumption.csv",
    header=0,
    infer_datetime_format=True,
    parse_dates=["datetime"],
    index_col=["datetime"],
)
# resample data to daily
daily_groups = dataset.resample("D")
daily_data = daily_groups.sum()
# summarize
print(daily_data.shape)
print(daily_data.head())
# save
daily_data.to_csv("household_power_consumption_days.csv")

# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    train, test = data[1:-328], data[-328:-6]
    # restructure into windows of weekly data
    train = array(split(train, len(train) / 7))
    test = array(split(test, len(test) / 7))
    return train, test


# load the new file
dataset = read_csv(
    "household_power_consumption_days.csv",
    header=0,
    infer_datetime_format=True,
    parse_dates=["datetime"],
    index_col=["datetime"],
)
train, test = split_dataset(dataset.values)
# validate train data
print(train.shape)
print(train[0, 0, 0], train[-1, -1, 0])
# validate test
print(test.shape)
print(test[0, 0, 0], test[-1, -1, 0])


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ", ".join(["%.1f" % s for s in scores])
    print("%s: [%.3f] %s" % (name, score, s_scores))


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)


# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 0, 70, 16
    n_timesteps, n_features, n_outputs = (
        train_x.shape[1],
        train_x.shape[2],
        train_y.shape[1],
    )
    # define model
    model = Sequential()
    model.add(LSTM(200, activation="relu", input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(n_outputs))
    model.compile(loss="mse", optimizer="adam")
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


# evaluate a single model
def evaluate_model(train, test, n_input):
    # fit model
    model = build_model(train, n_input)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores
