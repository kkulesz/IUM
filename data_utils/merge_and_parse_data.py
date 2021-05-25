from datetime import timedelta

import pandas as pd
import numpy as np


def parse_users(df):
    # Women - True ; Men - False
    df['gender'] = np.where(df['name'].str.split(' ').str[0].str[-1] == 'a', True, False)
    cities = df['city']
    encoded_cities = pd.get_dummies(cities.apply(pd.Series).stack()).sum(level=0)
    df = pd.concat([df, encoded_cities], axis=1)

    return df.drop(['name', 'street'], axis=1, inplace=False)


def parse_products(df):
    price_limit = 10000
    # get rid of rows with strange values
    df = df[(df.price < price_limit) & (df.price > 0)]

    # one-hot encode category path
    encoded = df['category_path'].str.split(';')
    encoded = pd.get_dummies(encoded.apply(pd.Series).stack()).sum(level=0)
    encoded.columns = encoded.columns.str.replace(' ', '_')
    encoded.columns = ['cat_' + str(col) for col in encoded.columns]
    df = pd.concat([df, encoded], axis=1)

    df = df.drop(['category_path'], axis=1, inplace=False)

    return df


def parse_sessions(df):
    # convert ids to int and get rid of null value rows
    df['user_id'] = df['user_id'].astype(float).astype('Int64')
    df['product_id'] = df['product_id'].astype(float).astype('Int64')

    df = df.dropna(subset=['user_id'])
    df = df.dropna(subset=['product_id'])

    df = df.drop(['purchase_id'], axis=1, inplace=False)

    return df


def increment_counter(user_id, sessions_dict):
    if user_id in sessions_dict:
        sessions_dict[user_id] += 1
    else:
        sessions_dict[user_id] = 0


def get_counter(user_id, sessions_dict):
    if user_id in sessions_dict:
        return sessions_dict[user_id]
    return 0


def get_counter_and_increment(user_id, sessions_dict):
    increment_counter(user_id, sessions_dict)
    return get_counter(user_id, sessions_dict)


def merge_dataframes(users, products, sessions):
    df = pd.merge(sessions, products, on='product_id', how='inner')

    df.sort_values(by=['timestamp'], ascending=True)
    session_ids = df['session_id'].unique()

    columns_list = df.columns.tolist()
    categories_columns = [col for col in columns_list if col.startswith('cat_')]

    # for how many sessions of user there were already
    successful_so_far_dict = {}
    sessions_so_far_dict = {}

    rows = []
    for s_id in session_ids:
        s = df[(df.session_id == s_id)]
        num_of_rows = s.shape[0]

        # skip events that took less that 2 seconds, it was probably a missclick
        s.sort_values('timestamp')
        for i in range(0, num_of_rows - 1):
            item_shown_time = s['timestamp'].iloc[i + 1] - s['timestamp'].iloc[i]
            if abs(item_shown_time) < timedelta(seconds=2):
                s.drop(index=i)

        # process single session rows into one
        length = s.timestamp.max() - s.timestamp.min()
        discount = s.offered_discount.unique()[0]
        user_id = s.user_id.unique()[0]
        successful = len(s[s.event_type == 'BUY_PRODUCT']) > 0

        mean_price = round(s.price.mean(), 2)
        median_of_price = round(s.price.median(), 2)

        min_rating = round(s.rating.min(), 2)
        max_rating = round(s.rating.max(), 2)
        mean_rating = round(s.rating.mean(), 2)

        # get user's history so far
        how_many_so_far = get_counter_and_increment(user_id, sessions_so_far_dict)
        how_many_successful_so_far = get_counter(user_id, successful_so_far_dict)
        if successful:
            increment_counter(user_id, successful_so_far_dict)
        percentage_of_successful = 0
        if how_many_so_far != 0:
            percentage_of_successful = round(how_many_successful_so_far / how_many_so_far, 2)

        # sum one-hot encoded
        summed_categories = s[categories_columns].sum().to_dict()

        # new_session = {'length': length, 'discount': discount,
        #                'user_id': user_id, 'successful': successful,
        #                'mean_price': mean_price, 'median_of_price': median_of_price,
        #                'min_rating': min_rating, 'max_rating': max_rating,
        #                'mean_rating': mean_rating, 'sessions_so_far': how_many_so_far,
        #                'successful_sessions_so_far': how_many_successful_so_far,
        #                }
        new_session = {'length': length, 'discount': discount,
                       'user_id': user_id, 'successful': successful,
                       'mean_price': mean_price, 'median_of_price': median_of_price,
                       'min_rating': min_rating, 'max_rating': max_rating,
                       'mean_rating': mean_rating,
                       'percentage_of_successful': percentage_of_successful
                       }

        new_session.update(summed_categories)
        rows.append(new_session)

    merged_sessions_df = pd.DataFrame(rows)
    merged_sessions_df = pd.merge(merged_sessions_df, users, on='user_id', how='inner')

    return merged_sessions_df


def read_and_parse_data():
    users_file_path = 'data/users.jsonl'
    products_file_path = 'data/products.jsonl'
    sessions_file_path = 'data/sessions.jsonl'

    users_df = pd.read_json(users_file_path, lines=True)
    products_df = pd.read_json(products_file_path, lines=True)
    sessions_df = pd.read_json(sessions_file_path, lines=True)

    users_df = parse_users(users_df)
    products_df = parse_products(products_df)
    sessions_df = parse_sessions(sessions_df)

    ready_data = merge_dataframes(users_df, products_df, sessions_df)
    return ready_data


def encode_boolean(b):
    if b is False:
        return 0
    else:
        return 1


def convert_length(length):
    days, time = length.split('days')
    days = int(days)
    seconds = days * 86400  # 1 day = 86400 s
    h, m, s = time.split(':')
    h, m, s = int(h), int(m), int(s)
    seconds += h * 3600
    seconds += m * 60
    seconds += s
    return seconds


def encode_and_save_data(df):
    df.drop(['city'], axis=1, inplace=True) #TODO: user_id_zostawic/wywalic
    df['length'] = df['length'].astype(str).apply(convert_length)
    df['successful'] = df['successful'].apply(encode_boolean)
    df['gender'] = df['gender'].apply(encode_boolean)
    df.to_csv('data/data.csv', index=None)

    # for data with no categories
    cols = [c for c in df.columns if c[:3] != 'cat']
    df = df[cols]
    df.to_csv('data/data_no_cats.csv', index=None)


if __name__ == '__main__':
    data = read_and_parse_data()
    encode_and_save_data(data)
