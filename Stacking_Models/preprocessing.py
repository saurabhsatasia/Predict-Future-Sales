import pandas as pd
import numpy as np

class Preprocess:
    def __init__(self):
        self.data_path = 'D:/Strive/Predict-Future-Sales/data'

    def load_join(self):
        items = pd.read_csv('data/items.csv', dtype={'item_name': 'str', 'item_id': 'int32',
                                                     'item_category_id': 'int32'})  # item_name, item_id, item_category_id
        items_t = pd.read_csv('data/items-translated.csv', dtype={'item_name': 'str',
                                                                  'item_id': 'int32'})  # item_name, item_id, item_category_id missing
        items_t['item_category_id'] = items['item_category_id']
        itm_cat = pd.read_csv('data/item_categories-translated.csv', dtype={'item_category_name': 'str',
                                                                            'item_category_id': 'int32'})  # item_category_name, item_category_id
        sales_train = pd.read_csv('data/sales_train.csv', parse_dates=['date'],
                                  dtype={'date': 'str', 'date_block_num': 'int32', 'shop_id': 'int32',
                                         'item_id': 'int32', 'item_price': 'float32',
                                         'item_cnt_day': 'int32'})  # date, date_block_num,shop_id,item_id,item_price, item_cnt_day
        shops_t = pd.read_csv('data/shops-translated.csv',
                              dtype={'shop_name': 'str', 'shop_id': 'int32'})  # shop_name, shop_id
        self.test = pd.read_csv('data/test.csv',
                           dtype={'ID': 'int32', 'shop_id': 'int32', 'item_id': 'int32'})  # ID,shop_id, item_id
        self.train = sales_train.join(items_t, on='item_id', rsuffix='_').join(shops_t, on='shop_id', rsuffix='_'). \
                            join(itm_cat, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'],axis=1)

        return self.test, self.train

    def data_leak(self, train_df, test_df):
        train_df = train_df.query('item_price > 0')
        test_shop_ids = test_df['shop_id'].unique()
        test_item_ids = test_df['item_id'].unique()
        # Only shops that exist in test set.
        lk_train = train_df[train_df['shop_id'].isin(test_shop_ids)]
        # Only items that exist in test set.
        lk_train = lk_train[lk_train['item_id'].isin(test_item_ids)]
        print('Data set size before leaking:', train_df.shape[0])
        print('Data set size after leaking:', lk_train.shape[0])
        return lk_train

    def