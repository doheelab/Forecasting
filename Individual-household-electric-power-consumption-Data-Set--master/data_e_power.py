# ### * The aim is just to show how to build the simplest Long short-term memory (LSTM) recurrent neural network for the data.
# ### The description of data can be found here:
# http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
#
# ### Attribute Information:
#
# #### 1.date: Date in format dd/mm/yyyy
# #### 2.time: time in format hh:mm:ss
# #### 3.global_active_power: household global minute-averaged active power (in kilowatt)
# #### 4.global_reactive_power: household global minute-averaged reactive power (in kilowatt)
# #### 5.voltage: minute-averaged voltage (in volt)
# #### 6.global_intensity: household global minute-averaged current intensity (in ampere)
# #### 7.sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
# #### 8.sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
# #### 9.sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.
#
# Let`s import all packages that we may need:

import sys
import numpy as np  # linear algebra
from scipy.stats import randint
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt  # this is used for the plot the graph
import seaborn as sns  # used for plot interactive graph.
from sklearn.model_selection import train_test_split  # to split the data into two parts
from sklearn.model_selection import KFold  # use for cross validation
from sklearn.preprocessing import StandardScaler  # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline  # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics  # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error, r2_score

## for Deep-learing:
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

df = pd.read_csv(
    "../../input/household_power_consumption.txt",
    sep=";",
    parse_dates={"dt": ["Date", "Time"]},
    infer_datetime_format=True,
    low_memory=False,
    na_values=["nan", "?"],
    index_col="dt",
)

# ### 1) Note that data include 'nan' and '?' as a string. I converted both to numpy nan in importing stage (above) and treated both of them the same.
# ### 2) I merged two columns 'Date' and 'Time' to 'dt'.
# ### 3) I also converted in the above, the data to time-series type, by taking index to be the time.

df.info()
df.describe()

# #  Dealing with missing values  'nan' with a test statistic
## finding all columns that have nan:

# filling nan with mean in any columns
for j in range(0, 7):
    df.iloc[:, j] = df.iloc[:, j].fillna(df.iloc[:, j].mean())

# another sanity check to make sure that there are not more any nan
df.isnull().sum()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# ### * In order to reduce the computation time, and also get a quick result to test the model.  One can resmaple the data over hour (the original data are given in minutes). This will reduce the size of data from 2075259 to 34589 but keep the overall strucure of data as shown in the above.

## resampling of data over hour
df_resample = df.resample("h").mean()
df_resample.shape

# ## * Note: I scale all features in range of [0,1].
## If you would like to train based on the resampled data (over hour), then used below
values = df_resample.values

## full data without resampling
# values = df.values

# integer encode direction
# ensure all data is float
# values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[8, 9, 10, 11, 12, 13]], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values

n_train_time = 365 * 24
train = values[:n_train_time, :]
test = values[n_train_time:, :]
##test = values[n_train_time:n_test_time, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# We reshaped the input into the 3D format as expected by LSTMs, namely [samples, timesteps, features].

# # Model architecture
#
# ### 1)  LSTM with 100 neurons in the first visible layer
# ### 3) dropout 20%
# ### 4) 1 neuron in the output layer for predicting Global_active_power.
# ### 5) The input shape will be 1 time step with 7 features.
#
# ### 6) I use the Mean Absolute Error (MAE) loss function and the efficient Adam version of stochastic gradient descent.
# ### 7) The model will be fit for 20 training epochs with a batch size of 70.
#
#

model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
#    model.add(LSTM(70))
#    model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

# fit network
history = model.fit(
    train_X,
    train_y,
    epochs=20,
    batch_size=70,
    validation_data=(test_X, test_y),
    verbose=2,
    shuffle=False,
)

# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper right")
plt.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], 7))

# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print("Test RMSE: %.3f" % rmse)

# ### Note that in order to improve the model, one has to adjust epochs and batch_size.

# # Final remarks
# ### * Here I have used the LSTM which is now the state-of-the-art for sequencial or time-series problems.
# ### * In order to reduce the computation time, and get some results quickly, I took the first year of data (resampled over hour) to train the model and the rest of data to test the model.  The above codes work for any time interval (just one has to change one line to change the interval).
# ### * I put together a very simple LSTM neural-network to show that one can obtain reasonable predictions.
# However numbers of rows is too high and as a result the computation is very time-consuming (even for the simple model in the above it took few mins to be run on  2.8 GHz Intel Core i7).  The Best is to write the last part of code using Spark (MLlib) running on GPU.
# ### * Moreover, the neural-network architecture that I have designed is a toy model.
# It can be easily improved by dropout and adding CNN layers.
# The CNN is useful here since there are correlations in data (CNN layer is a good way to probe the local structure of data).

# %%

