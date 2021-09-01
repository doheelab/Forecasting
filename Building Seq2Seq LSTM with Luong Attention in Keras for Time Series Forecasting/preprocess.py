import random
import numpy as np
from scipy.stats.stats import spearmanr
import matplotlib.pyplot as plt
import pickle as pkl

# # Create Data

n_ = 1000
t = np.linspace(0, 50 * np.pi, n_)
# pattern + trend + noise
x1 = (
    sum([20 * np.sin(i * t + np.pi) for i in range(5)])
    + 0.01 * (t ** 2)
    + np.random.normal(0, 6, n_)
)
x2 = (
    sum([15 * np.sin(2 * i * t + np.pi) for i in range(5)])
    + 0.5 * t
    + np.random.normal(0, 6, n_)
)

plt.figure(figsize=(15, 4))
plt.plot(range(len(x1)), x1, label="x1")
plt.plot(range(len(x2)), x2, label="x2")
plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2
)
plt.show()


plt.figure(figsize=(15, 4))
plt.hist(np.column_stack([x1, x2]).flatten(), bins=100, alpha=0.5)
plt.title("value distribution")
plt.show()

# # Prepare Data
# ### Split

train_ratio = 0.8
train_len = int(train_ratio * t.shape[0])
print(train_len)

# ### Detrend
x_index = np.array(range(len(t)))

# assume we already know the order of trend
x1_trend_param = np.polyfit(x_index[:train_len], x1[:train_len], 2)
x2_trend_param = np.polyfit(x_index[:train_len], x2[:train_len], 1)
print(x1_trend_param)
print(x2_trend_param)

x1_trend = (
    (x_index ** 2) * x1_trend_param[0] + x_index * x1_trend_param[1] + x1_trend_param[2]
)
x2_trend = x_index * x2_trend_param[0] + x2_trend_param[1]

plt.figure(figsize=(15, 4))
plt.plot(range(len(x1)), x1, label="x1")
plt.plot(range(len(x1_trend)), x1_trend, linestyle="--", label="x1_trend")

plt.plot(range(len(x2)), x2, label="x2")
plt.plot(range(len(x2_trend)), x2_trend, linestyle="--", label="x2_trend")
plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2
)
plt.show()

x1_detrend = x1 - x1_trend
x2_detrend = x2 - x2_trend

plt.figure(figsize=(15, 4))
plt.plot(range(len(x1_detrend)), x1_detrend, label="x2_detrend")
plt.plot(range(len(x2_detrend)), x2_detrend, label="x2_detrend")
plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2
)
plt.show()

# ### Combine
# x_lbl columns: ===================
# columns1: detrended x1,
# columns2: detrended x2,
# columns3: index
# columns4: 1 for train set, 0 for test set

x_lbl = np.column_stack(
    [
        x1_detrend,
        x2_detrend,
        x_index,
        [1] * train_len + [0] * (len(x_index) - train_len),
    ]
)
print(x_lbl.shape)
print(x_lbl)

# ### Normalize
x_train_max = x_lbl[x_lbl[:, 3] == 1, :2].max(axis=0)
x_train_max = x_train_max.tolist() + [1] * 2  # only normalize for the first 2 columns
print(x_train_max)

# x_lbl columns: ===================
# columns1: normalized detrended x1,
# columns2: normalized detrended x2,
# columns3: index
# columns4: 1 for train set, 0 for test set

x_normalize = np.divide(x_lbl, x_train_max)
print(x_normalize)

plt.figure(figsize=(15, 4))
plt.plot(range(train_len), x_normalize[:train_len, 0], label="x1_train_normalized")
plt.plot(range(train_len), x_normalize[:train_len, 1], label="x2_train_normalized")
plt.plot(
    range(train_len, len(x_normalize)),
    x_normalize[train_len:, 0],
    label="x1_test_normalized",
)
plt.plot(
    range(train_len, len(x_normalize)),
    x_normalize[train_len:, 1],
    label="x2_test_normalized",
)
plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2
)
plt.show()

# ### Truncate
def truncate(
    x,
    feature_cols=range(3),
    target_cols=range(3),
    label_col=3,
    train_len=100,
    test_len=20,
):
    in_, out_, lbl = [], [], []
    for i in range(len(x) - train_len - test_len + 1):
        in_.append(x[i : (i + train_len), feature_cols].tolist())
        out_.append(
            x[(i + train_len) : (i + train_len + test_len), target_cols].tolist()
        )
        lbl.append(x[i + train_len, label_col])
    return np.array(in_), np.array(out_), np.array(lbl)


X_in, X_out, lbl = truncate(
    x_normalize,
    feature_cols=range(3),
    target_cols=range(3),
    label_col=3,
    train_len=200,
    test_len=20,
)
print(X_in.shape, X_out.shape, lbl.shape)


X_input_train = X_in[np.where(lbl == 1)]
X_output_train = X_out[np.where(lbl == 1)]

X_input_test = X_in[np.where(lbl == 0)]
X_output_test = X_out[np.where(lbl == 0)]

print(X_input_train.shape, X_output_train.shape)
print(X_input_test.shape, X_output_test.shape)

pkl.dump(
    {
        "X_input_train": X_input_train,
        "X_output_train": X_output_train,
        "X_input_test": X_input_test,
        "X_output_test": X_output_test,
        "x1_trend_param": x1_trend_param,
        "x2_trend_param": x2_trend_param,
        "x_train_max": x_train_max,
    },
    open("data.pkl", "wb"),
)

