from datetime import datetime as dt
import pandas as pd
from data_preprocess import *
from lin_reg import *
import pytz


features = [
        'accommodates',
        'bathrooms',
        'longitude',
        'latitude',
        'beds',
        'bedrooms',
        'square_feet',
        'host_is_superhost',
        'room_type',
        'review_scores_rating',
        'zipcode',
        # 'host_listings_count',
        # 'number_of_reviews',
        'number_of_reviews_ltm',
        # 'review_scores_accuracy',
        # 'review_scores_cleanliness',
        # 'review_scores_checkin',
        # 'review_scores_communication',
        # 'review_scores_location',
        # 'review_scores_value',
        ]

# load main csv file with all data
# then clean it

full_df = data_preprocessing(pd.read_csv("data.csv"),features,normalization = 'scaling')

test_ind = pd.read_csv("test.csv")
train_ind = pd.read_csv("train.csv")
val_ind = pd.read_csv("val.csv")

train_df = full_df.merge(train_ind, on=["id"])
train_df_prices = train_df['price']
y_train = train_df_prices.values
train_df = train_df[features]
X_train = train_df.values

val_df = full_df.merge(val_ind, on=["id"])
val_df_prices = val_df['price']
y_val = val_df_prices.values
val_df = val_df[features]
X_val = val_df.values

test_df = full_df.merge(test_ind, on=["id"])
test_df = test_df[features]
X_test = test_df.values

lr =0.01
n_iter = 100000
reg = 0.01

theta = np.random.randn(len(train_df.columns) + 1,1)
X_b = np.c_[np.ones((len(X_train),1)),X_train]

y_train = y_train.reshape((-1,1))
theta = gradient_descent(X_b,y_train,theta,lr,n_iter,reg)

X_b_val = np.c_[np.ones((len(X_val),1)),X_val]
val_predict = X_b_val.dot(theta)
y_val = y_val.reshape((-1,1))
rmse = np.sqrt(np.mean(np.square(val_predict - y_val)))
print("RMSE Validation: %.3f" % rmse)

# X_b_test = np.c_[np.ones((len(X_test),1)),X_test]
# predicted_prices = X_b_test.dot(theta)
#
# tz_NY = pytz.timezone('America/New_York')
# datetime_NY = dt.now(tz_NY)
# time_stamp = datetime_NY.strftime("%H_%M_%S")
#
# answ_df = pd.read_csv("test.csv")
# answ_df.insert(1,'price',predicted_prices)
# answ_df.to_csv("predictions/%d_f%d_%s.csv" % (rmse,len(features),time_stamp),index=False)
#
# print("Finished")