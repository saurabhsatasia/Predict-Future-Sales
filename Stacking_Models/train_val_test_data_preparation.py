import pandas as pd
import numpy as np
"""
Train/validation split
-> As we know the test set in on the future, so we should try to simulate the same distribution on our train/validation split.
-> Our train set will be the first 3~28 blocks, validation will be last 5 blocks (29~32) and test will be block 33.
-> I'm leaving the first 3 months out because we use a 3 month window to generate features, so these first 3 month won't have really windowed useful features.
"""


class Data_Preparation:
    def __init__(self):
        pass

    def train_val_test_set(self, train_monthly_df):
        train_set = train_monthly_df.query('date_block_num >= 3 and date_block_num < 28').copy()
        validation_set = train_monthly_df.query('date_block_num >= 28 and date_block_num < 33').copy()
        test_set = train_monthly_df.query('date_block_num == 33').copy()
        # drop NaNs from item_cnt_month
        train_set.dropna(subset=['item_cnt_month'], inplace=True)
        validation_set.dropna(subset=['item_cnt_month'], inplace=True)

        train_set.dropna(inplace=True)
        validation_set.dropna(inplace=True)

        print('Train set records:', train_set.shape[0])
        print('Validation set records:', validation_set.shape[0])
        print('Test set records:', test_set.shape[0])

        print(f'Train set records: {train_set.shape[0]}. ({((train_set.shape[0] / train_monthly_df.shape[0]) * 100)}% of complete data)')
        print(f'Validation set records: {validation_set.shape[0]}. ({((validation_set.shape[0] / train_monthly_df.shape[0]) * 100)}% of complete data)')

        return train_set, validation_set, test_set

    def train_val_mean_encode(self, train_df, val_df):
        # Shop mean encoding.
        gp_shop_mean = train_df.groupby(['shop_id']).agg({'item_cnt_month': ['mean']})
        gp_shop_mean.columns = ['shop_mean']
        gp_shop_mean.reset_index(inplace=True)
        # Item mean encoding.
        gp_item_mean = train_df.groupby(['item_id']).agg({'item_cnt_month': ['mean']})
        gp_item_mean.columns = ['item_mean']
        gp_item_mean.reset_index(inplace=True)
        # Shop with item mean encoding.
        gp_shop_item_mean = train_df.groupby(['shop_id', 'item_id']).agg({'item_cnt_month': ['mean']})
        gp_shop_item_mean.columns = ['shop_item_mean']
        gp_shop_item_mean.reset_index(inplace=True)
        # Year mean encoding.
        gp_year_mean = train_df.groupby(['year']).agg({'item_cnt_month': ['mean']})
        gp_year_mean.columns = ['year_mean']
        gp_year_mean.reset_index(inplace=True)
        # Month mean encoding.
        gp_month_mean = train_df.groupby(['month']).agg({'item_cnt_month': ['mean']})
        gp_month_mean.columns = ['month_mean']
        gp_month_mean.reset_index(inplace=True)

        # Add mean encoding features to train set.
        train_df = pd.merge(train_df, gp_shop_mean, on=['shop_id'], how='left')
        train_df = pd.merge(train_df, gp_item_mean, on=['item_id'], how='left')
        train_df = pd.merge(train_df, gp_shop_item_mean, on=['shop_id', 'item_id'], how='left')
        train_df = pd.merge(train_df, gp_year_mean, on=['year'], how='left')
        train_df = pd.merge(train_df, gp_month_mean, on=['month'], how='left')
        # Add mean encoding features to validation set.
        val_df = pd.merge(val_df, gp_shop_mean, on=['shop_id'], how='left')
        val_df = pd.merge(val_df, gp_item_mean, on=['item_id'], how='left')
        val_df = pd.merge(val_df, gp_shop_item_mean, on=['shop_id', 'item_id'], how='left')
        val_df = pd.merge(val_df, gp_year_mean, on=['year'], how='left')
        val_df = pd.merge(val_df, gp_month_mean, on=['month'], how='left')
        # Create train and validation sets and labels.
        X_train = train_df.drop(['item_cnt_month', 'date_block_num'], axis=1)
        Y_train = train_df['item_cnt_month'].astype(int)
        X_validation = val_df.drop(['item_cnt_month', 'date_block_num'], axis=1)
        Y_validation = val_df['item_cnt_month'].astype(int)

        return X_train, Y_train, X_validation, Y_validation

    def prepare_test_data(self, test_set_df):
