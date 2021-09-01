import random
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    RepeatVector,
    TimeDistributed,
    Input,
    BatchNormalization,
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

import pydot as pyd
from tensorflow.keras.utils import plot_model
import tensorflow

tensorflow.keras.utils.pydot = pyd

# # Read Data
data = pkl.load(open("data.pkl", "rb"))

X_input_train = data["X_input_train"]
X_output_train = data["X_output_train"]
X_input_test = data["X_input_test"]
X_output_test = data["X_output_test"]

x1_trend_param = data["x1_trend_param"]
x2_trend_param = data["x2_trend_param"]
x_train_max = data["x_train_max"]

input_train = Input(shape=(X_input_train.shape[1], 1))
output_train = Input(shape=(X_output_train.shape[1], 1))


n_hidden = 30
print(input_train)
print(output_train)

# #### What to Return using LSTM
# - __*return_sequences=False, return_state=False*__: the last hidden state: state_h
# - __*return_sequences=True, return_state=False*__: return stacked hidden states (num_timesteps * num_cells): one hidden state output for each input time step
# - __*return_sequences=False, return_state=True*__: generate 3 arrays: state_h, state_h, state_c
# - __*return_sequences=True, return_state=True*__: generate 3 arrays: stacked hidden states, last state_h, last state_c

# output state : output corresponding to all timesteps,
# final_memory(hidden)_state : output corresponding to the last timestep,
# final_carray(cell)_state : last cell state

encoder_last_h1, _, encoder_last_c = LSTM(
    n_hidden,
    activation="elu",
    dropout=0.2,
    recurrent_dropout=0.2,
    return_sequences=False,
    return_state=True,
)(input_train)

encoder_last_h1 = BatchNormalization(momentum=0.6)(encoder_last_h1)
encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)
encoder_state = [encoder_last_h1, encoder_last_c]
decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)

decoder = LSTM(
    n_hidden,
    activation="elu",
    dropout=0.2,
    recurrent_dropout=0.2,
    return_state=False,
    return_sequences=True,
)(decoder, initial_state=encoder_state)

out = TimeDistributed(Dense(output_train.shape[2]))(decoder)

model = Model(inputs=input_train, outputs=out)
opt = Adam(lr=0.01, clipnorm=1)
model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mae"])
model.summary()


plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)


epc = 10  # 100
es = EarlyStopping(monitor="val_loss", mode="min", patience=10)
history = model.fit(
    X_input_train[:, :, 0],
    X_output_train[:, :, 0],
    # validation_split=0.2,
    validation_data=(X_input_test[:, :, 0], X_output_test[:, :, 0]),
    epochs=epc,
    verbose=1,
    callbacks=[es],
    batch_size=100,
)

train_mae = history.history["mae"]
valid_mae = history.history["val_mae"]

# model.save("save/model_forecasting_seq2seq.h5")
# model.load_weights("../save/model_forecasting_seq2seq.h5")


plt.plot(train_mae, label="train mae"),
plt.plot(valid_mae, label="validation mae")
plt.ylabel("mae")
plt.xlabel("epoch")
plt.title("train vs. validation accuracy (mae)")
plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2
)
plt.show()

result = model.predict(X_input_test[:, :, 0])

for idx in range(10):
    plt.plot(X_output_test[idx, :, 0])
    plt.plot(result[idx, :, 0])
    plt.show()


# X_input_train.shape
# X_input_test.shape

# X_input_test[:,:,0].shape

#########################

# # Predict

# %%
train_pred_detrend = model.predict(X_input_train[:, :, 0]) * x_train_max[0]
test_pred_detrend = model.predict(X_input_test[:, :, 0]) * x_train_max[0]
print(train_pred_detrend.shape, test_pred_detrend.shape)

train_true_detrend = X_output_train[:, :, 0] * x_train_max[0]
test_true_detrend = X_output_test[:, :, 0] * x_train_max[0]
print(train_true_detrend.shape, test_true_detrend.shape)


# combine with the index
train_pred_detrend = np.concatenate(
    [train_pred_detrend, np.expand_dims(X_output_train[:, :, 2], axis=2)], axis=2
)
test_pred_detrend = np.concatenate(
    [test_pred_detrend, np.expand_dims(X_output_test[:, :, 2], axis=2)], axis=2
)
print(train_pred_detrend.shape, test_pred_detrend.shape)

train_true_detrend = np.concatenate(
    [train_true_detrend, np.expand_dims(X_output_train[:, :, 2], axis=2)], axis=2
)
test_true_detrend = np.concatenate(
    [test_true_detrend, np.expand_dims(X_output_test[:, :, 2], axis=2)], axis=2
)
print(train_pred_detrend.shape, test_pred_detrend.shape)


# recover trend
data_final = dict()

for dt, lb in zip(
    [train_pred_detrend, train_true_detrend, test_pred_detrend, test_true_detrend],
    ["train_pred", "train_true", "test_pred", "test_true"],
):
    dt_x1 = (
        dt[:, :, 0]
        + (dt[:, :, 2] ** 2) * x1_trend_param[0]
        + dt[:, :, 2] * x1_trend_param[1]
        + x1_trend_param[2]
    )
    dt_x2 = dt[:, :, 1] + dt[:, :, 2] * x2_trend_param[0] + x2_trend_param[1]
    data_final[lb] = np.concatenate(
        [np.expand_dims(dt_x1, axis=2), np.expand_dims(dt_x2, axis=2)], axis=2
    )
    print(lb + ": {}".format(data_final[lb].shape))


for k in ["train_pred", "train_true", "test_pred", "test_true"]:
    print("maximum: {}".format(k))
    print(data_final[k].max())

# MAE train / MAE test
for lb in ["train", "test"]:
    MAE_overall = abs(data_final[lb + "_pred"] - data_final[lb + "_true"]).mean()
    MAE_ = abs(data_final[lb + "_pred"] - data_final[lb + "_true"]).mean(axis=(1, 2))
    plt.figure(figsize=(15, 3))
    plt.plot(MAE_)
    plt.title("MAE " + lb + ": overall MAE = " + str(MAE_overall))
    plt.show()

# nth sample 에서 train, test 실제/예측 결과 plot
for lb in ["train", "test"]:
    ith_sample = random.choice(range(data_final[lb + "_pred"].shape[0]))

    plt.figure(figsize=(15, 3))
    for i, x_lbl, clr in zip([0, 1], ["x1", "x2"], ["green", "blue"]):
        plt.plot(
            data_final[lb + "_pred"][ith_sample, :, i],
            linestyle="--",
            color=clr,
            label="pred " + x_lbl,
        )
        plt.plot(
            data_final[lb + "_true"][ith_sample, :, i],
            linestyle="-",
            color=clr,
            label="true " + x_lbl,
        )
    plt.title("({}): {}th sample".format(lb, ith_sample))
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=False,
        ncol=2,
    )
    plt.show()


# %%

# 모든 샘플에서 n번째 예측 결과 표출
ith_timestep = random.choice(range(data_final[lb + "_pred"].shape[1]))
for lb in ["train", "test"]:
    plt.figure(figsize=(15, 5))
    for i, x_lbl, clr in zip([0, 1], ["x1", "x2"], ["green", "blue"]):
        plt.plot(
            data_final[lb + "_pred"][:, ith_timestep, i],
            linestyle="--",
            color=clr,
            label="pred " + x_lbl,
        )
        plt.plot(
            data_final[lb + "_true"][:, ith_timestep, i],
            linestyle="-",
            color=clr,
            label="true " + x_lbl,
        )
    plt.title("({}): {}th time step in all samples".format(lb, ith_timestep))
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=False,
        ncol=2,
    )
    plt.show()

# %%
plt.figure(figsize=(15, 5))
train_start_t = 0
test_start_t = data_final["train_pred"].shape[0]
for lb, tm, clrs in zip(
    ["train", "test"],
    [train_start_t, test_start_t],
    [["green", "red"], ["blue", "orange"]],
):
    for i, x_lbl in zip([0, 1], ["x1", "x2"]):
        plt.plot(
            range(tm, tm + data_final[lb + "_pred"].shape[0]),
            data_final[lb + "_pred"][:, ith_timestep, i],
            linestyle="--",
            linewidth=1,
            color=clrs[0],
            label="pred " + x_lbl,
        )
        plt.plot(
            range(tm, tm + data_final[lb + "_pred"].shape[0]),
            data_final[lb + "_true"][:, ith_timestep, i],
            linestyle="-",
            linewidth=1,
            color=clrs[1],
            label="true " + x_lbl,
        )


plt.title("{}th time step in all samples".format(lb, ith_timestep))
plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=8
)
plt.show()

