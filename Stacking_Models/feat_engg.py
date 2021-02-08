import pandas as pd
import numpy as np

class Feature_Eng:
    def __init__(self):
        pass

    def add_features(self, train_monthly):
        # Unitary Price
        train_monthly['item_price_unit'] = train_monthly['item_price'] // train_monthly['item_cnt']
        train_monthly['item_price_unit'].fillna(0, inplace=True)
        # Group based features
        gp_item_price = train_monthly.sort_values('date_block_num').groupby(['item_id'], as_index=False).agg(
            {'item_price': [np.min, np.max]})
        gp_item_price.columns = ['item_id', 'hist_min_item_price', 'hist_max_item_price']
        train_monthly = pd.merge(train_monthly, gp_item_price, on='item_id', how='left')

        # How much each item's price changed from its (lowest/highest) historical price.
        train_monthly['price_increase'] = train_monthly['item_price'] - train_monthly['hist_min_item_price']
        train_monthly['price_decrease'] = train_monthly['hist_max_item_price'] - train_monthly['item_price']
        return train_monthly

    def roll_window_features(self, train_monthly):
        # Min value
        f_min = lambda x: x.rolling(window=3, min_periods=1).min()
        # Max value
        f_max = lambda x: x.rolling(window=3, min_periods=1).max()
        # Mean value
        f_mean = lambda x: x.rolling(window=3, min_periods=1).mean()
        # Standard deviation
        f_std = lambda x: x.rolling(window=3, min_periods=1).std()

        function_list = [f_min, f_max, f_mean, f_std]
        function_name = ['min', 'max', 'mean', 'std']

        for i in range(len(function_list)):
            train_monthly[('item_cnt_%s' % function_name[i])] = \
            train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_category_id', 'item_id'])[
                'item_cnt'].apply(function_list[i])

        # Fill the empty std features with 0
        train_monthly['item_cnt_std'].fillna(0, inplace=True)
        return train_monthly

    def lag_based_features(self, train_monthly):
        lag_list = [1, 2, 3]
        for lag in lag_list:
            ft_name = ('item_cnt_shifted%s' % lag)
            train_monthly[ft_name] = \
            train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_category_id', 'item_id'])[
                'item_cnt'].shift(lag)
            # Fill the empty shifted features with 0
            train_monthly[ft_name].fillna(0, inplace=True)
        return  train_monthly

    def item_sales_cnt_trend(self, train_monthly):
        lag_list = [1, 2, 3]
        train_monthly['item_trend'] = train_monthly['item_cnt']

        for lag in lag_list:
            ft_name = ('item_cnt_shifted%s' % lag)
            train_monthly['item_trend'] -= train_monthly[ft_name]

        train_monthly['item_trend'] /= len(lag_list) + 1

        return train_monthly

