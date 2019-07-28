import sys
sys.path.append('../')
import pandas as pd                     
import numpy as np                     
import datetime as dt                     
import os                     

class Indicators(object):
    
    #indicadores tecnicos:
    #-Media movel simples SMA
    #bandas de bollinger BB
    #indice de forca relativa RSI
    #Momentum
    
    def __init__(self, data, window = 21):
        
        self.window = window
        self.data = data
        self.rolling = data.rolling( window )

    def get_return(self):
    
        return self.data.pct_change()
    
    def get_rolling_mean(self, **kwargs):
        
        self.rolling.window = kwargs.get('window') if kwargs else self.window
        
        return self.rolling.mean()
    

    def get_rolling_std(self, **kwargs):
        
        self.rolling.window = kwargs.get('window') if kwargs else self.window
        
        return self.rolling.std()

    def get_bollinger_bands(self, rm, rstd):
        
        upper_band = rm + 2*rstd
       
        lower_band = rm - 2*rstd
       
        return upper_band, lower_band
    
    def get_rolling_mean_value(self, sma, normalized = False ):

        mean_value = self.data/sma - 1
        
        return self.normalize(mean_value) if normalized else mean_value
        
    
    def get_bollinger_bands_value(self, sma, rstd, normalized = False):
        
        bands_value = (self.data - sma)/(2 * rstd)
        
        return self.normalize(bands_value) if normalized else bands_value
    
    def get_momentum_value(self, n = 5, normalized = False):
        
        momentum_value = self.data/self.data.shift(n) - 1 
        
        return self.normalize(momentum_value) if normalized else momentum_value
    
    def get_rsi_value(self, window = 14, normalized = False):
        #obs: primeiro valor do df eh contado como donw_loss
        window += 1
        up_gain = (self.data.diff() >= 0).rolling(window).sum()
        donw_loss = (self.data.diff() < 0).rolling(window).sum()
        rs = (up_gain/window)/(donw_loss/window)
        
        rsi = 1*(1 - 1/( 1 + rs))
        
        return self.normalize(rsi) if normalized else rsi
        
    
    def normalize(self, data):
        
        return (data - data.mean())/data.std()


def conc_df(dataframes, names = None):
#         Recebe os dataframes e os concatenam
#        
#         inputs:
#             dataframes: list (df1,df2,...,dfn)
#             names: list[str] (optional) 
#         output: exemplo
#               df1  =                    GOOG       SPY       IBM
#                         2008-01-02  1.000000  1.000000  1.000000
#                         2008-01-03  1.000204  0.999546  1.002080
#                         2008-01-04  0.958858  0.975028  0.965987
#                         2008-01-07  0.947547  0.974196  0.955690
#                         2008-01-08  0.921905  0.958456  0.932182

#                 df2  =                  GOOG       SPY       IBM
#                         2008-01-02       NaN       NaN       NaN
#                         2008-01-03  1.000102  0.999773  1.001040
#                         2008-01-04  0.979531  0.987287  0.984034
#                         2008-01-07  0.953203  0.974612  0.960838
#                         2008-01-08  0.934726  0.966326  0.943936

#     conc_df([df1,df2['SPY']],names = ['price','SMA']) = 
#                                    price                           SMA
#                                     GOOG       SPY       IBM       SPY
#                     2008-01-02  1.000000  1.000000  1.000000       NaN
#                     2008-01-03  1.000204  0.999546  1.002080  0.999773
#                     2008-01-04  0.958858  0.975028  0.965987  0.987287
#                     2008-01-07  0.947547  0.974196  0.955690  0.974612
#                     2008-01-08  0.921905  0.958456  0.932182  0.966326
    
    names = names if len(names) == len(dataframes) else ['df_' + str(i) for i in range(len(dataframes))]
    
    return pd.concat(dataframes, axis=1, keys=names)


def test_code():                     
 

    Symbol = ['GOOG','IBM','SPY']               
    sv = 1e5                     
    
    dev_sd=dt.datetime(2007,1,1)
    dev_ed=dt.datetime(2009,12,31)
    test_sd=dt.datetime(2010,1,1)
    test_ed=dt.datetime(2011,12,31)
    
    start_date, end_date = dev_sd, dev_ed 
                     
   
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(Symbol, dates,False)
    
    prices_all = fill_missing_values(prices_all)
    
    prices_all = prices_all/prices_all.iloc[0]
    
    i = Indicators(prices_all, window= 21)
    
    sma = i.get_rolling_mean()

    sma_41 = i.get_rolling_mean( window = 41)

    sma_66 = i.get_rolling_mean( window = 66)
    
    rstd = i.get_rolling_std()
    
    upper,lower = i.get_bollinger_bands(sma, rstd)
    
    normalized = False
    
    bnds_value = i.get_bollinger_bands_value(sma, rstd, normalized = normalized)
    
    mean_value = i.get_rolling_mean_value( sma, normalized= normalized )

    mean_value_41 = i.get_rolling_mean_value( sma_41, normalized= normalized )

    mean_value_66 = i.get_rolling_mean_value( sma_66, normalized= normalized )
    
    momentum_value = i.get_momentum_value( n = 5, normalized= normalized)
    
    rsi_value = i.get_rsi_value()
    
    names = ['price','sma','upper','lower',
            'bnds_value','mean_value','momentum_value','rsi_value',
            'sma_41','sma_66','mean_value_41','mean_value_66']
    
    
    for i in Symbol:
        
        dfs = conc_df([prices_all[[i]],
                       sma[i],upper[i],
                       lower[i],
                       bnds_value[i],
                      mean_value[i],
                      momentum_value[i],
                       rsi_value[i],
                       sma_41[i],
                       sma_66[i],
                       mean_value_41[i],
                       mean_value_66[i]
                      ],names = names)
        fig = graph(dfs)
      
        plot(fig,filename = "Time Series with Rangeslider")
    
    
if __name__ == '__main__':
    
    test_code()