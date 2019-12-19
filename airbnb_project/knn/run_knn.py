import numpy as np


def get_weights(x):
    """returns weights based on how far a listing is away from point"""
    y = 1.0 - 1.0 / (1 + np.exp(-20*np.power(x,2)))
    y = y/np.sum(y)
    return y

def lin_weights(x):
    """linear weights based on how far listing is away from a point"""
    sum = np.sum(x)
    return x / sum

def predict_price_multivariate(new_listing,feature_columns,train_df=None, k = 8):
    
    def compute_eucl_dist(new_listing,train_df,list_of_feautures):
        sum_squares = 0
        for feature in list_of_feautures:
            sum_squares += np.square(train_df[feature] - new_listing[feature])
            
        return np.sqrt(sum_squares)

    # compute Euclidean distances for all listings and find k closest ones
    train_df['distance'] = compute_eucl_dist(new_listing, train_df, feature_columns)
    knn_df = train_df.sort_values('distance')
    knn = knn_df.iloc[:k]

    # get weights and predict price by weighted sum of prices
    weights = get_weights(knn['distance'].values)
    predicted_price = np.sum(knn['price'].values * weights)

    return predicted_price