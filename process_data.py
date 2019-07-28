import pandas as pd
import numpy as np
from indicators import Indicators


class FeatureExtractor:
    def __init__(self, df):
        self.df = df
        self.open = df['open'].astype('float')
        self.close = df['close'].astype('float')
        self.high = df['high'].astype('float')
        self.low = df['low'].astype('float')
        self.volume = df['volume'].astype('float')
        self.indicators = Indicators(self.close, window=21)

    def add_features(self, normalized=False):
        # stationary candle
        sma = self.indicators.get_rolling_mean()
        sma_41 = self.indicators.get_rolling_mean(window=41)
        sma_66 = self.indicators.get_rolling_mean(window=66)
        rstd = self.indicators.get_rolling_std()
        self.df['ret_day'] = self.indicators.get_return()
        self.df['bnds_value'] = self.indicators.get_bollinger_bands_value(sma, rstd, normalized=normalized)
        self.df['mean_value'] = self.indicators.get_rolling_mean_value(sma, normalized=normalized)
        self.df['mean_value_41'] = self.indicators.get_rolling_mean_value(sma_41, normalized=normalized)
        self.df['mean_value_66'] = self.indicators.get_rolling_mean_value(sma_66, normalized=normalized)
        self.df['momentum_value'] = self.indicators.get_momentum_value(n=5, normalized=normalized)
        self.df['rsi_value'] = self.indicators.get_rsi_value()
        return self.df 
