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

        itm_cat = pd.read_csv('data/item_categories-translated.csv', dtype={'item_category_name': 'str', 'item_category_id': 'int32'})  # item_category_name, item_category_id
        sales_train = pd.read_csv('data/sales_train.csv', parse_dates=['date'],
                                  dtype={'date': 'str', 'date_block_num': 'int32', 'shop_id': 'int32',
                                         'item_id': 'int32', 'item_price': 'float32',
                                         'item_cnt_day': 'int32'})  # date, date_block_num,shop_id,item_id,item_price, item_cnt_day
        shops_t = pd.read_csv('data/shops-translated.csv', dtype={'shop_name': 'str', 'shop_id': 'int32'})  # shop_name, shop_id
        self.test = pd.read_csv('data/test.csv', dtype={'ID': 'int32', 'shop_id': 'int32', 'item_id': 'int32'})  # ID,shop_id, item_id
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

    def create_montly_df(self, lk_train_df):
        train_monthly = lk_train_df[['date', 'date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'item_cnt_day']]

        # Group by month in this case "date_block_num" and aggregate features.
        train_monthly = train_monthly.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_category_id', 'item_id'], as_index=False)
        train_monthly = train_monthly.agg({'item_price': ['sum', 'mean'], 'item_cnt_day': ['sum', 'mean', 'count']})
        train_monthly.columns = ['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price',
                                 'mean_item_price', 'item_cnt', 'mean_item_cnt', 'transactions']
        # Build a data set with all the possible combinations of ['date_block_num','shop_id','item_id'] so we won't have missing records.
        shop_ids = train_monthly['shop_id'].unique()
        item_ids = train_monthly['item_id'].unique()
        empty_df = []
        for i in range(34):
            for shop in shop_ids:
                for item in item_ids:
                    empty_df.append([i, shop, item])

        empty_df = pd.DataFrame(empty_df, columns=['date_block_num', 'shop_id', 'item_id'])  # each repeating shop_id has multiple unique item_id
        # Merge the train set with the complete set (missing records will be filled with 0).
        train_monthly = pd.merge(empty_df, train_monthly, on=['date_block_num', 'shop_id', 'item_id'], how='left')
        train_monthly.fillna(0, inplace=True)
        return train_monthly

    def create_year_month(self, train_monthly_df):
        # Extract time based features.
        train_monthly_df['year'] = train_monthly_df['date_block_num'].apply(lambda x: ((x // 12) + 2013))
        train_monthly_df['month'] = train_monthly_df['date_block_num'].apply(lambda x: (x % 12))
        return train_monthly_df

    def remove_outliers(self, train_monthly_df):
        # item_cnt > 20 and < 0, item_price >= 400000 as outliers, so remove them.
        train_monthly_df.query('item_cnt >= 0 and item_cnt <= 20 and item_price < 400000')
        # Creating the label
        # Our label will be the "item_cnt" of the next month, as we are dealing with a forecast problem.
        train_monthly_df['item_cnt_month'] = train_monthly_df.sort_values('date_block_num').groupby(['shop_id', 'item_id'])['item_cnt'].shift(-1)
        return train_monthly_df