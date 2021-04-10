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

    # TODO: przeróbka kategorii, skłaniałbym się do rozbicia ich i wrzucenia do one-hot-encoding

    return df


def parse_sessions(df):
    # convert ids to int and get rid of null value rows
    df['user_id'] = df['user_id'].astype(float).astype('Int64')
    df['product_id'] = df['product_id'].astype(float).astype('Int64')

    df = df.dropna(subset=['user_id'])
    df = df.dropna(subset=['product_id'])

    df = df.drop(['purchase_id'], axis=1, inplace=False)
    print(df.head(20))
    return df


def merge_dataframes(users, products, sessions):
    # session_ids = df['session_id'].unique()
    '''
    TODO:
    1. merge sesji i produktów na product_id
    2. TODO: dokonczyc todo ;p

    '''

    # rows = []
    # for ses_id in session_ids:
    #     s = df[(df.session_id == ses_id)]
    #     length = s.timestamp.max() - s.timestamp.min()
    #     discount = s.offered_discount.unique()[0]
    #     user_id = s.user_id.unique()[0]
    #     successful = len(s[s.event_type == 'BUY_PRODUCT']) > 0
    #     seen = s.product_id.tolist()
    #     bought = s.loc[
    #         (s.event_type == 'BUY_PRODUCT'), 'product_id'].tolist()
    #
    #     new_session = {'length': length, 'discount': discount,
    #                    'user_id': user_id, 'successful': successful,
    #                    'seen': seen, 'bought': bought}
    #
    #     rows.append(new_session)
    #
    # merged_sessions_df = pd.DataFrame(rows)
    # print(merged_sessions_df.info())
    pass


def read_and_parse_data():
    users_file_path = 'data/users.jsonl'
    products_file_path = 'data/products.jsonl'
    sessions_file_path = 'data/sessions.jsonl'

    users_df = pd.read_json(users_file_path, lines=True)
    products_df = pd.read_json(products_file_path, lines=True)
    sessions_df = pd.read_json(sessions_file_path, lines=True)

    # users_df = parse_users(users_df)
    # products_df = parse_products(products_df)
    # sessions_df = parse_sessions(sessions_df)

    ready_data = merge_dataframes(users_df, products_df, sessions_df)
    #TODO: zapisać i korzystać ;)

if __name__ == '__main__':
    read_and_parse_data()
