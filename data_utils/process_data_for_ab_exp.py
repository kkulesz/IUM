import pandas as pd


def divide_users(data):
    users = set(data['user_id'])
    even_users = [user for user in users if int(user) % 2 == 0]
    odd_users = list(users.difference(even_users))
    return even_users, odd_users


def divide_data(data, groups):
    a_data = data.loc[data['user_id'].isin(groups[0])]
    b_data = data.loc[data['user_id'].isin(groups[1])]
    return a_data, b_data


if __name__ == '__main__':
    d = pd.read_csv('data/data_with_user_id.csv')
    groups = divide_users(d)
    divide_data(d, groups)
    data_a, data_b = divide_data(d, groups)
    data_a.to_csv('data/data_a.csv', index=None)
    data_b.to_csv('data/data_b.csv', index=None)
