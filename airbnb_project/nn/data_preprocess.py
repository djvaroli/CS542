import numpy as np


def data_preprocessing(full_df,features,normalization = '0-means'):
    # seperate features into groups for processing
    print("Doing data cleaning!")
    continuous = ['bathrooms', 'beds', 'bedrooms','square_feet','host_listings_count','number_of_reviews','number_of_reviews_ltm',
                  'review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication',
                  'review_scores_location','review_scores_value']
    binary = ['host_is_superhost','host_identity_verified','is_location_exact'] 
    other = ['property_type','room_type','host_response_time']
    datetime = ['host_since','first_review','last_review']

    # for dt_f in datetime:
    #     full_d
    #     dt.strptime(, format_str))

    for cf in continuous:
        cf_median = np.nanmedian(full_df[cf])
        full_df[cf].fillna(cf_median,inplace=True)

    for bf in binary:
        full_df[bf].fillna('f',inplace=True)

    rsr_median = np.nanmedian(full_df.review_scores_rating)
    full_df.review_scores_rating.fillna(rsr_median, inplace=True)

    # create key value dictionaries to use for replacing strings with floats
    binary_dict = {'t':1,'f':0} 
    room_type_dict = {'Entire home/apt':3, 'Private room':1, 'Hotel room':2, 'Shared room':0}
    property_type_dict = {'House':1.5,'Apartment':1, 'Other': 0}
    response_rate_dict = {'within an hour':2.0, 'within a day':1.0, 'within a few hours':1.5,
       'a few days or more':0}

    dicts = {**binary_dict,**room_type_dict,**property_type_dict,**response_rate_dict}

    full_df.property_type.replace({
        'Townhouse': 'House',
        'Serviced apartment': 'Apartment',
        'Loft': 'Apartment',
        'Bungalow': 'House',
        'Cottage': 'House',
        'Villa': 'House',
        'Tiny house': 'House',
        'Earth house': 'House',
        'Chalet': 'House'  
        }, inplace=True)

    # replace all property types not in the above dict with other
    full_df.loc[~full_df.property_type.isin(['House', 'Apartment']), 'property_type'] = 'Other'

    full_df.replace(binary_dict,inplace=True)
    full_df.replace(room_type_dict,inplace=True)
    full_df.replace(property_type_dict, inplace=True)


    #### AMENITIES #####
    key_amenities = ['Free parking on premises','Free street parking','Patio or balcony','Gym','Pool','Air conditioning']

    # score the amenities by weight
    chars_to_strip = {"'","{","}",'"'}
    am_entries = full_df.amenities
    scores = np.zeros(len(am_entries))

    i = 0
    for entry in am_entries:
        for char_to_strip in chars_to_strip:
            entry = entry.replace(char_to_strip,'')

        entry = entry.split(",")
        score = 0

        for amenity in entry:
            if amenity in key_amenities:
                amenity_weight = 1
            else:
                amenity_weight = 1

            score += amenity_weight

        scores[i] = score
        i += 1

    full_df['amenities'] = list(scores)

    ##### ZIPCODE PROCESSING ######
    zipcodes_df = full_df.zipcode

    zipcodes = np.zeros(len(zipcodes_df))

    iZip = 0
    for zipcode in zipcodes_df:
        if type(zipcode) == str:
            zipcode = zipcode.replace('TX','')
            zipcode = zipcode.replace(' ','')
            zipcodes[iZip]= float(zipcode)
        else:
            zipcodes[iZip] = np.nan
        iZip += 1

    full_df['zipcode'] = list(zipcodes)
    full_df.zipcode.fillna(np.nanmin(full_df.zipcode), inplace = True)


    def normalize(full_df,normalization = '0-mean'):
        if normalization == '0-mean':
            for f in features:
                f_mean = np.mean(full_df[f])
                f_std = np.std(full_df[f])
                full_df[f] = (full_df[f] - f_mean)/f_std

        elif normalization == 'scaling':
            for f in features:
                f_max = np.max(full_df[f])
                f_min = np.min(full_df[f])
                full_df[f] = (full_df[f] - f_min)/(f_max - f_min)

        elif normalization == '-1to1':
            for f in features:
                f_max = np.max(full_df[f])
                f_min = np.min(full_df[f])
                full_df[f] = (full_df[f] - f_min)/(f_max - f_min) * (1 + 1) - 1

        return full_df

    
    print(normalization)
    full_df = normalize(full_df,normalization)
    
    return full_df
