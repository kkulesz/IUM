import pandas as pd


def encode_boolean(b):
    if b is False:
        return 0
    else:
        return 1


def convert_length(leng):
    days, time = leng.split('days')
    days = int(days)
    seconds = days * 86400  # 1 day = 86400 s
    h, m, s = time.split(':')
    h, m, s = int(h), int(m), int(s)
    seconds += h * 3600
    seconds += m * 60
    seconds += s
    return seconds


if __name__ == '__main__':
    data = pd.read_csv('data/parsed_data.csv')
    data.drop(data.columns[[0]], axis=1, inplace=True)
    data = data.drop(['user_id', 'city'], axis=1)
    data['length'] = data['length'].apply(convert_length)
    data['successful'] = data['successful'].apply(encode_boolean)
    data['gender'] = data['gender'].apply(encode_boolean)
    data.to_csv('data/data.csv', index=None)

    # cols = [c for c in data.columns if c[:3] != 'cat']
    # data = data[cols]
    # data.to_csv('data/data_no_cats.csv', index=None)
