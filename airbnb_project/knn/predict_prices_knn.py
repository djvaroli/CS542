from datetime import datetime as dt
import pytz
import pandas as pd
from os.path import join
from data_preprocess import *
from run_knn import *


# combination of features that yielded best results
features = [
        'accommodates',
        'bathrooms',
        'longitude',
        'latitude',
        'beds',
        'bedrooms',
        'host_is_superhost',
        'room_type',
        'review_scores_rating',
         'host_listings_count',
         'number_of_reviews_ltm',
        ]

# load main csv file with all data
# then clean it
data_file = join("data","data.csv")
test_file = join("data","test.csv")
train_file = join("data","train.csv")
val_file = join("data","val.csv")

full_df = data_preprocessing(pd.read_csv(data_file),features,normalization = 'scaling')

# seperate into test,train and validation data

test_ind = pd.read_csv(test_file)
train_ind = pd.read_csv(train_file)
val_ind = pd.read_csv(val_file)

train_df = full_df.merge(train_ind, on=["id"])
train_df = train_df.sample(frac=1).reset_index(drop=True)

val_df = full_df.merge(val_ind, on=["id"])
val_df = val_df.sample(frac=1).reset_index(drop=True)
test_df = full_df.merge(test_ind, on=["id"])

### optimize for k over a range ####
k_range = 15
k_vals = np.arange(k_range,k_range + 1)
rmse_history = []

for k_ in k_vals:
        val_df['predicted_price'] = val_df[features].apply(predict_price_multivariate,feature_columns=features,train_df = train_df, k=k_,axis=1)
        rmse = np.sqrt(np.mean(np.square(val_df['predicted_price'] - val_df['price'])))
        rmse_history.append(rmse)

# get the k_value which yields the best RMSE
arg_min = np.argmin(rmse_history)
k_opt = k_vals[arg_min]

# since using KNN, use train + validation for price prediction
large_train = [train_df,val_df]
large_train_df = pd.concat(large_train,sort=True)
large_train_df = large_train_df.sample(frac=1).reset_index(drop=True)
train_df = large_train_df

# predict the prices
predicted_prices = test_df[features].apply(predict_price_multivariate,feature_columns=features,train_df = train_df,k = k_opt,axis=1)

# creating a time_stamp so that I could identify between the submission csv files
# write the prediction to a csv file
# tz_NY = pytz.timezone('America/New_York')
# datetime_NY = dt.now(tz_NY)
# time_stamp = datetime_NY.strftime("%H_%M_%S")

answ_df = pd.read_csv(test_file)
answ_df.insert(1,'price',predicted_prices)
answ_df.to_csv("pred.csv",index=False)
print("Finished")

