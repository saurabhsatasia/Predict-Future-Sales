import pandas as pd
import numpy as np

class Feature_Eng:
    def __init__(self):
        pass

    def unit_price_item(self, train_monthly):
        # Unitary Price
        train_monthly['item_price_unit'] = train_monthly['item_price'] // train_monthly['item_cnt']
        train_monthly['item_price_unit'].fillna(0, inplace=True)
        