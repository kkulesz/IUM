import pandas as pd
import numpy as np


def parse_users(df):
    # Women - True ; Men - False
    df['gender'] = np.where(df['name'].str.split(' ').str[0].str[-1] == 'a', True, False)

    return df.drop(['name', 'street'], axis=1, inplace=False)


def parse_products(df):
    price_limit = 10000
    # get rid of rows with strange values
    df = df[(df.price < price_limit) & (df.price > 0)]

    # one-got encode category path
    encoded = df['category_path'].str.split(';')
    encoded = pd.get_dummies(encoded.apply(pd.Series).stack()).sum(level=0)
    encoded.columns = encoded.columns.str.replace(' ', '_')
    encoded.columns = [ 'cat_' + str(col) for col in encoded.columns]
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
    # print(df.head(20))
    return df


def merge_dataframes(users, products, sessions):
    df = pd.merge(sessions, products, on='product_id', how='inner')
    # df = pd.merge(df, users, on='user_id', how='inner')


    session_ids = df['session_id'].unique()

    columns_list = df.columns.tolist()
    categories_columns = [col for col in columns_list if col.startswith('cat_')]

    rows = []
    for s_id in session_ids:
        s = df[(df.session_id == s_id)]

        # process single session rows into one
        length = s.timestamp.max() - s.timestamp.min()
        discount = s.offered_discount.unique()[0]
        user_id = s.user_id.unique()[0]
        successful = len(s[s.event_type == 'BUY_PRODUCT']) > 0
        mean_price = s.price.mean()

        # sum one-hot encoded
        summed_categories = s[categories_columns].sum().to_dict()

        new_session = {'length': length, 'discount': discount,
                       'user_id': user_id, 'successful': successful,
                       'mean_price': mean_price}

        new_session.update(summed_categories)

        rows.append(new_session)

    merged_sessions_df = pd.DataFrame(rows)
    merged_sessions_df = pd.merge(merged_sessions_df, users, on='user_id', how='inner')
    print(merged_sessions_df.info())
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

    ready_data.to_csv('data/merged_data.csv')

if __name__ == '__main__':
    read_and_parse_data()
