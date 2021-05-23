import requests
import pandas as pd

'''
simple script to send data to our app in order to log them and then analyze
'''

if __name__ == "__main__":
    app_url = 'http://127.0.0.1:5000'

    data = pd.read_csv('../data_utils/data/data_no_cats.csv')
    no_of_rows = data.shape[0]

    user_id = 100
    session_id = 5000
    for row_index in range(0, no_of_rows):
        prediction_data = data.loc[row_index, data.columns != 'successful']
        prediction_data['user_id'] = user_id
        prediction_data['session_id'] = session_id

        result_data = data.loc[row_index, data.columns == 'successful']
        result_data['successful'] = str(result_data['successful'] == 1.0)
        result_data['session_id'] = session_id

        json_to_prediction = prediction_data.to_dict()
        json_to_log_result = result_data.to_dict()

        #  send data to predict, and then send how the session actually ended
        requests.get(app_url + '/predict', json=json_to_prediction)
        requests.post(app_url + '/session_result', json=json_to_log_result)

        if row_index % 1000 == 0:
            print(f"[{row_index:4.0f}/{no_of_rows}]")

        user_id += 1
        session_id += 1

