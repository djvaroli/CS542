from data_preprocess import *
from os.path import join
from datetime import datetime as dt
import pandas as pd
from mlp_5 import *

import pytz

features = [
        'accommodates',
        'bathrooms',
        'longitude',
        'latitude',
        ]

data_file = join("data","data.csv")
full_df = data_preprocessing(pd.read_csv(data_file),features,normalization = '-1to1')


test_file = join("data","test.csv")
train_file = join("data","train.csv")
val_file = join("data","val.csv")

# seperate into test,train and validation data

test_ind = pd.read_csv(test_file)
train_ind = pd.read_csv(train_file)
val_ind = pd.read_csv(val_file)

cols = features

train_df = full_df.merge(train_ind, on=["id"])
train_df_prices = train_df['price']
y_train = train_df_prices.values
train_df = train_df[cols]
X_train = train_df.values

val_df = full_df.merge(val_ind, on=["id"])
val_df_prices = val_df['price']
y_val = val_df_prices.values
val_df = val_df[cols]
X_val = val_df.values

test_df = full_df.merge(test_ind, on=["id"])
test_df = test_df[cols]
X_test = test_df.values

input_size = len(cols)
print(X_train.shape)
print(input_size)
hidden_size = 4096
hidden_size_1 = 256
hidden_size_2 = 256
hidden_size_3 = 512
output_size = 1
NN = TwoLayerMLP_5(input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size, activation='relu')

np.random.seed(0)
relu_stats = NN.train(X_train, y_train, X_val,y_val,
                            num_epochs=100, batch_size = 25,
                            learning_rate=3.5e-3, learning_rate_decay=0.9,
                            reg=1e-10, verbose=True)

print('Training Complete!')