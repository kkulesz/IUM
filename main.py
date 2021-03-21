import pandas as pd



def read_and_parse_data():
    # file_path = 'dane/users.jsonl'

    # file_path = 'dane/products.jsonl'
        # trzeba podzielić kategorie na kolumny
        # wywalić wiesze gdzie cena <=0 i jakaś pojebanie wielka cena


    file_path = 'dane/sessions.jsonl'
        # trzeba wywalić wiersze gdzie PRODUCT_ID = null

    # file_path = 'dane/deliveries.jsonl'
        # to jest wgl niepotrzebne chyba

    df = pd.read_json(file_path, lines=True)



    #print(df.info())

    print(df['event_type'].unique())
    # print(df[''].unique())

if __name__ == '__main__':
    read_and_parse_data()