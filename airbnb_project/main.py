# main file
import pandas as pd
import numpy as np

full_df = pd.read_csv("data/data.csv")
features = list(full_df)

test_ind = pd.read_csv("data/test.csv")
train_ind = pd.read_csv("data/train.csv")
val_ind = pd.read_csv("data/val.csv")

train_df = full_df.merge(train_ind, on=["id"])
val_df = full_df.merge(val_ind, on=["id"])
test_df = full_df.merge(test_ind, on=["id"])

def predict_price(new_listing_value,feature_column,k = 5):
    temp_df = train_df
    temp_df['distance'] = np.abs(train_df[feature_column] - new_listing_value)
    temp_df = temp_df.sort_values('distance')
    knn = temp_df.price.iloc[:k]
    predicted_price = knn.mean()
    return predicted_price

prices = test_df.accommodates.apply(predict_price,feature_column='accommodates')


def predict_price_multivariate(new_listing_value, feature_columns):
    temp_df = norm_train_df
    temp_df['distance'] = distance.cdist(temp_df[feature_columns], [new_listing_value[feature_columns]])
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return (predicted_price)
    cols = ['accommodates', 'bathrooms']

norm_test_df['predicted_price'] = norm_test_df[cols].apply(predict_price_multivariate,feature_columns=cols,axis=1)
norm_test_df['squared_error'] = (norm_test_df['predicted_price'] - norm_test_df['price'])**(2)
mse = norm_test_df['squared_error'].mean()
rmse = mse ** (1/2)
print(rmse)
df_1 = pd.read_csv("data/test.csv")
df_1.insert(1,'price',prices)
df_1.to_csv("data/prediction.csv",index=False)


# df.replace() # create dictionary

# df['price'] = df.price.str.replace("\$|,",'').astype(float)

# df['price'].fillna(0,inplace=True)
