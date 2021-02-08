from Stacking_Models.preprocessing import Preprocess
from Stacking_Models.feat_engg import Feature_Eng

# Preprocessing
prepro = Preprocess()
test, train = prepro.load_join()
lk_train = prepro.data_leak(train_df=train, test_df=test)
train_monthly = prepro.create_montly_df(lk_train_df=lk_train)
train_monthly = prepro.create_montly_df(lk_train_df=train_monthly)
train_monthly = prepro.remove_outliers(train_monthly_df=train_monthly)

# Feature Engineering
fe = Feature_Eng()
train_monthly = fe.add_features(train_monthly=train_monthly)
train_monthly = fe.roll_window_features(train_monthly=train_monthly)
train_monthly = fe.lag_based_features(train_monthly=train_monthly)
train_monthly = fe.item_sales_cnt_trend(train_monthly=train_monthly)

# Train/Validation Split
