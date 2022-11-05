# Version Wintermute 20220314

import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import colorama
from colorama import Fore
#import talib as tape

#%matplotlib inline
pd.options.display.max_rows = 99999

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__) # necessaria versione >= 1.9.0

import cufflinks as cf

# Per utilizzo con Notebooks
init_notebook_mode(connected=True)

# Per utilizzo offline
cf.go_offline()

import warnings
warnings.filterwarnings("ignore")


# LOADING FUNCTIONS ********************************************************************************************************start

def load_data_intraday_old(filename: str) -> pd.core.frame.DataFrame:
    """
    Funzione per il parsing di una serie intraday 
    con estensione txt esportata da Tradestation
    """
    
    import datetime
    start = datetime.datetime.now()

    data = pd.read_csv(filename, 
                       usecols=['Date','Time','Open','High','Low','Close','Up','Down'], 
                       parse_dates=[['Date', 'Time']], )
    data.columns = ["date_time","open","high","low","close","up","down"]
    data.set_index('date_time', inplace = True)
    data['volume'] = data['up'] + data['down']
    data.drop(['up','down'],axis=1,inplace=True)
    data["dayofweek"] = data.index.dayofweek
    data["day"] = data.index.day
    data["month"] = data.index.month
    data["year"] = data.index.year
    data["dayofyear"] = data.index.dayofyear
    data["quarter"] = data.index.quarter
    data["hour"] = data.index.hour
    data["minute"] = data.index.minute
    
    end = datetime.datetime.now()
    print("loaded", len(data), "records in", end - start)
    
    return data

def load_data_intraday(filename):
    """
    Funzione per il parsing di una serie intraday 
    con estensione txt esportata da Tradestation
    """
    
    import datetime
    start = datetime.datetime.now()

    data = pd.read_csv(filename, 
                       usecols=['Date','Time','Open','High','Low','Close','Up','Down'], 
                       parse_dates=[['Date', 'Time']])
    data.columns = ["date_time","open","high","low","close","up","down"]
    data.set_index('date_time', inplace = True)
    data['volume'] = data['up'] + data['down']
    data.drop(['up','down'],axis=1,inplace=True)
    data["dayofweek"] = data.index.dayofweek
    data["day"] = data.index.day
    data["month"] = data.index.month
    data["year"] = data.index.year
    data["dayofyear"] = data.index.dayofyear
    data["quarter"] = data.index.quarter
    data["hour"] = data.index.hour
    data["minute"] = data.index.minute
    
    data["daily_open"] = daily_open(data,0)
    data["daily_open1"] = daily_open(data,1)
    data["daily_high1"] = daily_high(data,1)
    data["daily_low1"] = daily_low(data,1)
    data["daily_close1"] = daily_close(data,1)
    
    end = datetime.datetime.now()
    print("loaded", len(data), "records in", end - start)
    
    return data.dropna()

def load_data_daily(filename):
    """
    Funzione per il caricamento di uno storico daily
    Fonte dati: Tradestation .txt
    """
    
    import datetime
    start = datetime.datetime.now()
    
    data = pd.read_csv(filename, parse_dates = ["Date","Time"])
    data.columns = ["date","time","open","high","low","close","volume","oi"]
    data.set_index("date", inplace = True)
    data.drop(["time","oi"], axis=1, inplace=True)
    data["dayofweek"] = data.index.dayofweek
    data["day"] = data.index.day
    data["month"] = data.index.month
    data["year"] = data.index.year
    data["dayofyear"] = data.index.dayofyear
    data["quarter"] = data.index.quarter
    
    end = datetime.datetime.now()
    print("loaded", len(data), "records in", end - start)
    
    return data

def load_data_ffn(filename):
    """
    Function to load and elaborate an history 
    coming from FFN 
    """
    import datetime
    start = datetime.datetime.now()
    
    data = pd.read_csv(filename, usecols=['date','open','high','low','close','volume'])
    data.volume = data.volume.apply(lambda x: int(x))
    data["date"] = pd.to_datetime(data["date"])
    data.set_index('date', inplace = True)
    data["dayofweek"] = data.index.dayofweek
    data["day"] = data.index.day
    data["month"] = data.index.month
    data["year"] = data.index.year
    data["dayofyear"] = data.index.dayofyear
    data["quarter"] = data.index.quarter
    
    end = datetime.datetime.now()
    print("loaded", len(data), "records in", end - start)
    
    return data

def load_data_daily_rci(filename):
    """
    Funzione per il caricamento di uno storico daily
    Fonte dati: Gandalf Project Crypto .txt
    """
    
    import datetime
    start = datetime.datetime.now()
    
    data = pd.read_csv(filename, sep = " ", header = None)
    data.columns = ["date","open","high","low","close","volume"]
    data.set_index("date", inplace = True)
    data.index = pd.to_datetime(data.index, format='%d/%m/%Y')
    data["dayofweek"] = data.index.dayofweek
    data["day"] = data.index.day
    data["month"] = data.index.month
    data["year"] = data.index.year
    data["dayofyear"] = data.index.dayofyear
    data["quarter"] = data.index.quarter
    
    end = datetime.datetime.now()
    print("loaded", len(data), "records in", end - start)
    
    return data

def load_data_daily_ccxt(filename):
    """
    Function to load and elaborate an history 
    coming from CCXT Download 
    """
    
    import datetime
    start = datetime.datetime.now()
    
    data = pd.read_csv(filename)
    data.columns = ["date","open","high","low","close","volume"]
    data.volume = data.volume.apply(lambda x: int(x))
    data["date"] = pd.to_datetime(data["date"])
    data.set_index('date', inplace = True)
    #data.drop(['date'],axis=1,inplace=True)
    data["dayofweek"] = data.index.dayofweek
    data["day"] = data.index.day
    data["month"] = data.index.month
    data["year"] = data.index.year
    data["dayofyear"] = data.index.dayofyear
    data["quarter"] = data.index.quarter
    
    end = datetime.datetime.now()
    print("loaded", len(data), "records in", end - start)
    
    return data

def load_multiple_data_intraday(*args):
        
    i = 1
    # First of all we create aligned dataframe with all historical data
    for file in args:
        service = load_data_intraday(file)
        if i == 1:
            dataset = service
        else:
            dataset = pd.concat([dataset, service], axis = 1).dropna()
        i += 1
            
    fields = int(len(dataset.columns) / len(args))
    #print(fields)
        
    # Then we separate each historical data aligned
    results = []
    for i in range(1,len(args) + 1):
        if i == 1:
            results.append(dataset.iloc[:,:fields])
        else:
            results.append(dataset.iloc[:,(i - 1) * fields: i * fields])
            
    if len(results) == 1:
        return results[0]
    else:
        return results
    
def load_data_daily_slim(folder,filename):
    """
    Funzione per il caricamento di uno storico daily
    Fonte dati: Tradestation .txt
    """
    path = folder + filename
    data = pd.read_csv(path, parse_dates = ["Date","Time"])
    data.columns = ["date","time","open","high","low","close","volume","oi"]
    data.set_index("date", inplace = True)
    data.drop(["time","oi"], axis=1, inplace=True)
    
    return data

def load_multiple_data_daily(folder,asset_filelist):
        
    i = 1
    # First of all we create aligned dataframe with all historical data
    for file in asset_filelist:
        ticker = file.split("_")[0].lower()
        service = load_data_daily_slim(folder,file)
        new_col_names = []
        [new_col_names.append(ticker + "_" + col) for col in service.columns]
        service.columns = new_col_names 
        if i == 1:
            dataset = service
        else:
            dataset = pd.concat([dataset, service], axis = 1).dropna()
        i += 1

    return dataset

# LOADING FUNCTIONS **********************************************************************************************************end


# INDICATORS ***************************************************************************************************************start

def atr(data,period):
    """
    Function to calculate average true range
    Inputs: dataframe of prices ["open","high","low","close"]
    Output: average true range
    """
    data["m1"] = data.high - data.low
    data["m2"] = abs(data.high - data.close.shift(1))
    data["m3"] = abs(data.low - data.close.shift(1))
    data["maximum"] = data[["m1", "m2", "m3"]].max(axis = 1)
    data["atr"] = data.maximum.rolling(5).mean()
    return data.atr

def avg_true_range(dataframe, period):
    dataframe["M1"] = dataframe.high - dataframe.low
    dataframe["M2"] = abs(dataframe.high - dataframe.low.shift(1)).fillna(0)
    dataframe["M3"] = abs(dataframe.low - dataframe.close.shift(1)).fillna(0)
    dataframe["Max"] = dataframe[["M1", "M2", "M3"]].max(axis = 1)
    dataframe["MeanMax"] = dataframe["Max"].rolling(period).mean()
    return dataframe.MeanMax.fillna(0)

def RSI(series, period):
    """
    Function to calculate the Relative Strength Index of a close serie
    """
    df = pd.DataFrame(series, index = series.index)
    df["chg"] = series.diff(1)
    df["gain"] = np.where(df.chg > 0, 1, 0)
    df["loss"] = np.where(df.chg <= 0, 1, 0)
    df["avg_gain"] = df.gain.rolling(period).sum() / period * 100
    df["avg_loss"] = df.loss.rolling(period).sum() / period * 100
    rs = abs(df["avg_gain"] / df["avg_loss"])
    rsi = 100 - (100 / (1 + rs))
    return rsi

def MFI(df, period):
    """
    Function to calculate the Money Flow Index of a price serie (OHLCV)
    """
    df["typical_price"] = (df.iloc[:,1] + df.iloc[:,2] + df.iloc[:,3]) / 3
    df["raw_money_flow"] = df.typical_price * df.iloc[:,4]
    df["chg"] = df.raw_money_flow.diff(1)
    df["pos_money_flow"] = np.where(df.chg > 0,1,0)
    df["neg_money_flow"] = np.where(df.chg <= 0,1,0)
    df["avg_gain"] = df.pos_money_flow.rolling(period).sum() / period * 100
    df["avg_loss"] = df.neg_money_flow.rolling(period).sum() / period * 100
    mfr = abs(df["avg_gain"] / df["avg_loss"])
    mfi = 100 - (100 / (1 + mfr))
    return mfi

def BollingerBand(serie,period,multiplier):
    return serie.rolling(period).mean() + multiplier * serie.rolling(period).std()

def ROC(serie,period):
    return ((serie / serie.shift(period)) - 1) * 100

def Momentum(serie,period):
    return (serie - serie.shift(period))

def SMA(serie,period):
    return serie.rolling(period).mean()

def Range(serie):
    """
    Function to calculate the Range of a price serie (OHLCV)
    """
    return (serie.iloc[:,1] - serie.iloc[:,2])

def Body(serie):
    """
    Function to calculate the Body of a price serie (OHLCV)
    """
    return (serie.iloc[:,3] - serie.iloc[:,0])

def AvgPrice(serie):
    """
    Function to calculate the AvgPrice of a price serie (OHLCV)
    """
    return (serie.iloc[:,0] + serie.iloc[:,1] + serie.iloc[:,2] + serie.iloc[:,3]) / 4

def MedPrice(serie):
    """
    Function to calculate the MedPrice of a price serie (OHLCV)
    """
    return (serie.iloc[:,1] + serie.iloc[:,2]) / 2

def MedBodyPrice(serie):
    """
    Function to calculate the MedBodyPrice of a price serie (OHLCV)
    """
    return (serie.iloc[:,0] + serie.iloc[:,3]) / 2

def Blastoff(serie):
    return abs(serie.iloc[:,0] - serie.iloc[:,3]) / (serie.iloc[:,1] - serie.iloc[:,2])

def OpenToLowLog(dataset):
    data = dataset.copy()
    return pd.Series(np.log(data.open) - np.log(data.low))

def OpenToAvgPriceLog(dataset):
    data = dataset.copy()
    data["avgprice"] = data.iloc[:,:4].mean(axis = 1)
    return pd.Series(np.log(data.open) - np.log(data.avgprice))

def OpenToMedPriceLog(dataset):
    data = dataset.copy()
    data["medprice"] = (data.high + data.low) / 2
    return pd.Series(np.log(data.open) - np.log(data.medprice))

def OpenToOpenLog(dataset):
    data = dataset.copy()
    return pd.Series(np.log(data.open) - np.log(data.open.shift(1)))

def OpenToMedBodyPriceLog(dataset):
    data = dataset.copy()
    data["medbodyprice"] = (data.open + data.close) / 2
    return pd.Series(np.log(data.open) - np.log(data.medbodyprice))

def CloseToMedPriceLog(dataset):
    data = dataset.copy()
    data["medprice"] = (data.high + data.low) / 2
    return pd.Series(np.log(data.close) - np.log(data.medprice))

def CloseToCloseLog(dataset):
    data = dataset.copy()
    return pd.Series(np.log(data.close) - np.log(data.close.shift(1)))

def CloseToLowLog(dataset):
    data = dataset.copy()
    return pd.Series(np.log(data.close) - np.log(data.low))

def CloseToAvgPriceLog(dataset):
    data = dataset.copy()
    data["avgprice"] = data.iloc[:,:4].mean(axis = 1)
    return pd.Series(np.log(data.close) - np.log(data.avgprice))

def CloseToMedBodyPriceLog(dataset):
    data = dataset.copy()
    data["medbodyprice"] = (data.open + data.close) / 2
    return pd.Series(np.log(data.close) - np.log(data.medbodyprice))

def HighToCloseLog(dataset):
    data = dataset.copy()
    return pd.Series(np.log(data.high) - np.log(data.close))

def HighToOpenLog(dataset):
    data = dataset.copy()
    return pd.Series(np.log(data.high) - np.log(data.open))

def HighToMedBodyPriceLog(dataset):
    data = dataset.copy()
    data["medbodyprice"] = (data.open + data.close) / 2
    return pd.Series(np.log(data.high) - np.log(data.medbodyprice))

def HighToMedPriceLog(dataset):
    data = dataset.copy()
    data["medprice"] = (data.high + data.low) / 2
    return pd.Series(np.log(data.high) - np.log(data.medprice))

def HighToAvgPriceLog(dataset):
    data = dataset.copy()
    data["avgprice"] = data.iloc[:,:4].mean(axis = 1)
    return pd.Series(np.log(data.high) - np.log(data.avgprice))

def HighToHighLog(dataset):
    data = dataset.copy()
    return pd.Series(np.log(data.high) - np.log(data.high.shift(1)))

def LowToLowLog(dataset):
    data = dataset.copy()
    return pd.Series(np.log(data.low) - np.log(data.low.shift(1)))

def AvgPriceToMedPriceLog(dataset):
    data = dataset.copy()
    data["avgprice"] = data.iloc[:,:4].mean(axis = 1)
    data["medprice"] = (data.high + data.low) / 2
    return pd.Series(np.log(data.avgprice) - np.log(data.medprice))

def AvgPriceToLowLog(dataset):
    data = dataset.copy()
    data["avgprice"] = data.iloc[:,:4].mean(axis = 1)
    return pd.Series(np.log(data.avgprice) - np.log(data.low))

def AvgPriceToAvgPriceLog(dataset):
    data = dataset.copy()
    data["avgprice"] = data.iloc[:,:4].mean(axis = 1)
    return pd.Series(np.log(data.avgprice) - np.log(data.avgprice.shift(1)))

def AvgPriceToMedBodyPriceLog(dataset):
    data = dataset.copy()
    data["avgprice"] = data.iloc[:,:4].mean(axis = 1)
    data["medbodyprice"] = (data.open + data.close) / 2
    return pd.Series(np.log(data.avgprice) - np.log(data.medbodyprice))

def MedPriceToLowLog(dataset):
    data = dataset.copy()
    data["medprice"] = (data.high + data.low) / 2
    return pd.Series(np.log(data.medprice) - np.log(data.low))

def MedBodyPriceToLowLog(dataset):
    data = dataset.copy()
    data["medbodyprice"] = (data.open + data.close) / 2
    return pd.Series(np.log(data.medbodyprice) - np.log(data.low))

def MedPriceToMedBodyPriceLog(dataset):
    data = dataset.copy()
    data["medprice"] = (data.high + data.low) / 2
    data["medbodyprice"] = (data.open + data.close) / 2
    return pd.Series(np.log(data.medprice) - np.log(data.medbodyprice))

def MedPriceToMedPriceLog(dataset):
    data = dataset.copy()
    data["medprice"] = (data.high + data.low) / 2
    return pd.Series(np.log(data.medprice) - np.log(data.medprice.shift(1)))

def MedBodyPriceToMedBodyPriceLog(dataset):
    data = dataset.copy()
    data["medbodyprice"] = (data.open + data.close) / 2
    return pd.Series(np.log(data.medbodyprice) - np.log(data.medbodyprice).shift(1))

def RangeLog(dataset):
    data = dataset.copy()
    return pd.Series(np.log(data.high) - np.log(data.low))

def BodyLog(dataset):
    data = dataset.copy()
    return pd.Series(np.log(data.close) - np.log(data.open))

def BodyToRangeLog(dataset):
    data = dataset.copy()
    data["body"] = (data.close - data.open) 
    data["range"] = (data.high - data.low) 
    return pd.Series(np.log(data.body) - np.log(data.range))

def daily_high(dataframe, delay):
    df = dataframe.copy()
    df["rule"] = np.where(df["day"] != df["day"].shift(1),1,0)
    df["grouper"] = df.rule.cumsum()
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["high"].groupby(df["grouper"]).max().shift(delay))
    service.set_index(indexes, inplace=True)  
    field_name = "daily_high" + "_" + str(delay)
    service.columns = [field_name]
    df1 = pd.concat([df, service], axis = 1)
    df1[field_name] = df1[field_name].fillna(method = "ffill")
    return df1[field_name]

def weekly_high(dataframe, delay):
    df = dataframe.copy()
    df["rule"] = np.where(df["week"] != df["week"].shift(1),1,0)
    df["grouper"] = df.rule.cumsum()
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["high"].groupby(df["grouper"]).max().shift(delay))
    service.set_index(indexes, inplace=True)  
    field_name = "daily_high" + "_" + str(delay)
    service.columns = [field_name]
    df1 = pd.concat([df, service], axis = 1)
    df1[field_name] = df1[field_name].fillna(method = "ffill")
    return df1[field_name]
    
def daily_low(dataframe, delay):
    df = dataframe.copy()
    df["rule"] = np.where(df["day"] != df["day"].shift(1),1,0)
    df["grouper"] = df.rule.cumsum()
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["low"].groupby(df["grouper"]).min().shift(delay))
    service.set_index(indexes, inplace=True)  
    field_name = "daily_low" + "_" + str(delay)
    service.columns = [field_name]
    df1 = pd.concat([df, service], axis = 1)
    df1[field_name] = df1[field_name].fillna(method = "ffill")
    return df1[field_name]

def weekly_low(dataframe, delay):
    df = dataframe.copy()
    df["rule"] = np.where(df["week"] != df["week"].shift(1),1,0)
    df["grouper"] = df.rule.cumsum()
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["low"].groupby(df["grouper"]).min().shift(delay))
    service.set_index(indexes, inplace=True)  
    field_name = "daily_low" + "_" + str(delay)
    service.columns = [field_name]
    df1 = pd.concat([df, service], axis = 1)
    df1[field_name] = df1[field_name].fillna(method = "ffill")
    return df1[field_name]
    
def daily_open(dataframe, delay):
    df = dataframe.copy()
    df["rule"] = np.where(df["day"] != df["day"].shift(1),1,0)
    df["grouper"] = df.rule.cumsum()
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["open"].groupby(df["grouper"]).first().shift(delay))
    service.set_index(indexes, inplace=True)  
    field_name = "daily_open" + "_" + str(delay)
    service.columns = [field_name]
    df1 = pd.concat([df, service], axis = 1)
    df1[field_name] = df1[field_name].fillna(method = "ffill")
    return df1[field_name]

def weekly_open(dataframe, delay):
    df = dataframe.copy()
    df["rule"] = np.where(df["week"] != df["week"].shift(1),1,0)
    df["grouper"] = df.rule.cumsum()
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["open"].groupby(df["grouper"]).first().shift(delay))
    service.set_index(indexes, inplace=True)  
    field_name = "daily_open" + "_" + str(delay)
    service.columns = [field_name]
    df1 = pd.concat([df, service], axis = 1)
    df1[field_name] = df1[field_name].fillna(method = "ffill")
    return df1[field_name]
    
def daily_close(dataframe, delay):
    df = dataframe.copy()
    df["rule"] = np.where(df["day"] != df["day"].shift(1),1,0)
    df["grouper"] = df.rule.cumsum()
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["close"].groupby(df["grouper"]).last().shift(delay))
    service.set_index(indexes, inplace=True)  
    field_name = "daily_close" + "_" + str(delay)
    service.columns = [field_name]
    df1 = pd.concat([df, service], axis = 1)
    df1[field_name] = df1[field_name].fillna(method = "ffill")
    return df1[field_name]

def weekly_close(dataframe, delay):
    df = dataframe.copy()
    df["rule"] = np.where(df["week"] != df["week"].shift(1),1,0)
    df["grouper"] = df.rule.cumsum()
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["close"].groupby(df["grouper"]).last().shift(delay))
    service.set_index(indexes, inplace=True)  
    field_name = "daily_close" + "_" + str(delay)
    service.columns = [field_name]
    df1 = pd.concat([df, service], axis = 1)
    df1[field_name] = df1[field_name].fillna(method = "ffill")
    return df1[field_name]

def daily_close_sma(dataframe, delay, period):
    df = dataframe.copy()
    df["rule"] = np.where(df["day"] != df["day"].shift(1),1,0)
    df["grouper"] = df.rule.cumsum()
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["close"].groupby(df["grouper"]).last().shift(delay))
    service.set_index(indexes, inplace=True)  
    field_name = "daily_close" + "_" + str(delay)
    service.columns = [field_name]
    sma_field_name = "daily_close_sma" + "_" + str(delay)
    service[sma_field_name] = service[field_name].rolling(period).mean()
    df1 = pd.concat([df, service], axis = 1)
    df1[sma_field_name] = df1[sma_field_name].fillna(method = "ffill")
    return df1[sma_field_name]

def session_high(dataframe, session_hour, session_minute, delay):
    df = dataframe.copy()
    df["rule"] = np.where((df["hour"] == session_hour) & (df["minute"] == session_minute),1,0)
    df["grouper"] = df.rule.cumsum()
    df = df[df.grouper > 0] #innesto per cancellare i record precedenti al primo trigger
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["high"].groupby(df["grouper"]).max().shift(delay))#.dropna()
    service.set_index(indexes, inplace = True)  
    field_name = "session_high" + "_" + str(delay)
    service.columns = [field_name]
    df1 = pd.concat([df, service], axis = 1)
    df1[field_name] = df1[field_name].fillna(method = "ffill")
    return df1[field_name]

def session_low(dataframe, session_hour, session_minute, delay):
    df = dataframe.copy()
    df["rule"] = np.where((df["hour"] == session_hour) & (df["minute"] == session_minute),1,0)    
    df["grouper"] = df.rule.cumsum()
    df = df[df.grouper > 0] #innesto per cancellare i record precedenti al primo trigger
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["low"].groupby(df["grouper"]).min().shift(delay))
    service.set_index(indexes, inplace=True)  
    field_name = "session_low" + "_" + str(delay)
    service.columns = [field_name]
    df1 = pd.concat([df, service], axis = 1)
    df1[field_name] = df1[field_name].fillna(method = "ffill")
    return df1[field_name]

def session_open(dataframe, session_hour, session_minute, delay):
    df = dataframe.copy()
    df["rule"] = np.where((df["hour"] == session_hour) & (df["minute"] == session_minute),1,0)  
    df["grouper"] = df.rule.cumsum()
    df = df[df.grouper > 0] #innesto per cancellare i record precedenti al primo trigger
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["open"].groupby(df["grouper"]).first().shift(delay))
    service.set_index(indexes, inplace=True)  
    field_name = "session_open" + "_" + str(delay)
    service.columns = [field_name]
    df1 = pd.concat([df, service], axis = 1)
    df1[field_name] = df1[field_name].fillna(method = "ffill")
    return df1[field_name]
       
def session_close(dataframe, session_hour, session_minute, delay):
    df = dataframe.copy()
    df["rule"] = np.where((df["hour"] == session_hour) & (df["minute"] == session_minute),1,0) 
    df["grouper"] = df.rule.cumsum()
    df = df[df.grouper > 0] #innesto per cancellare i record precedenti al primo trigger
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["close"].groupby(df["grouper"]).last().shift(delay))
    service.set_index(indexes, inplace=True)  
    field_name = "session_close" + "_" + str(delay)
    service.columns = [field_name]
    df1 = pd.concat([df, service], axis = 1)
    df1[field_name] = df1[field_name].fillna(method = "ffill")
    return df1[field_name]

def session_close_sma(dataframe, session_hour, session_minute, delay, period):
    df = dataframe.copy()
    df["rule"] = np.where((df["hour"] == session_hour) & (df["minute"] == session_minute),1,0) 
    df["grouper"] = df.rule.cumsum()
    df = df[df.grouper > 0] #innesto per cancellare i record precedenti al primo trigger
    indexes = df[df.rule == 1].index
    service = pd.DataFrame(df["close"].groupby(df["grouper"]).last().shift(delay))
    service.set_index(indexes, inplace=True)  
    field_name = "session_close" + "_" + str(delay)
    service.columns = [field_name]
    sma_field_name = "session_close_sma" + "_" + str(delay)
    service[sma_field_name] = service[field_name].rolling(period).mean()
    df1 = pd.concat([df, service], axis = 1)
    df1[sma_field_name] = df1[sma_field_name].fillna(method = "ffill")
    return df1[sma_field_name]

def adx(data,period):
    """
    Function to calculate average directional index for a period of 14 days
    Inputs: dataframe of prices ["open","high","low","close"]
    Output: adx
    """
    data["+DM"] = np.where((data.high - data.high.shift(1) > data.low - data.low.shift(1)) & (data.high - data.high.shift(1)>0),\
                          data.high - data.high.shift(1),0)
    data["-DM"] = np.where((data.high - data.high.shift(1) < data.low - data.low.shift(1)) & (data.low - data.low.shift(1)>0),\
                          data.low - data.low.shift(1),0)
    data["m1"] = data.high - data.low
    data["m2"] = abs(data.high - data.close.shift(1))
    data["m3"] = abs(data.low - data.close.shift(1))
    data["TR"] = data[["m1", "m2", "m3"]].max(axis = 1)
    data["+DM14"] = data["+DM"].rolling(period).sum()
    data["-DM14"] = data["-DM"].rolling(period).sum()
    data["TR14"] = data["TR"].rolling(period).sum()
    data["+DI"] = round(((data["+DM14"]/data["TR14"])*100), 0)
    data["-DI"] = round(((data["-DM14"]/data["TR14"])*100), 0)
    data["DX"] = round(abs((data["+DI"]-data["-DI"])/(data["+DI"]+data["-DI"])),0)
    data["ADX"] = (data.DX.rolling(14).mean())*100
    return data.ADX

def trix(data,period):
    """
    Function to calculate triple exponential moving average
    Inputs: dataframe of prices ["open","high","low","close"]
    Output: trix
    """
    data["MOV_1"] = data.close.ewm(span=period, adjust=False).mean()
    data["MOV_2"] = data.MOV_1.ewm(span=period, adjust=False).mean()
    data["MOV_3"] = data.MOV_2.ewm(span=period, adjust=False).mean()
    data["TRIX"] = ((data.MOV_3 - data.MOV_3.shift(1))/data.MOV_3.shift(1))*100
    return data.TRIX

# INDICATORS *****************************************************************************************************************end

# PERFORMANCE METRICS ******************************************************************************************************start

def drawdown(equity):
    """
    Funzione che calcola il draw down data un'equity line
    """
    maxvalue = equity.expanding(0).max()
    drawdown = equity - maxvalue
    drawdown_series = pd.Series(drawdown, index = equity.index)
    return drawdown_series

def max_drawdown_date(equity):
    return drawdown(equity).idxmin()

def drawdown_perc(equity,basic_quantity):
    """
    Funzione che calcola il draw down percentuale data un'equity line 
    ed un capitale iniziale
    """
    real_equity = basic_quantity + equity
    eq_max = real_equity.expanding().max()
    dd = drawdown(equity)
    dd_perc = (dd / eq_max) * 100
    dd_perc = pd.Series(np.where(dd_perc > 0, 0 , dd_perc), index = equity.index)
    return dd_perc

def max_draw_down_perc(equity,basic_quantity):
    dd_perc = drawdown_perc(equity,basic_quantity)
    return round(dd_perc.min(),2)

def max_drawdown_perc_date(equity,basic_quantity):
    return drawdown_perc(equity,basic_quantity).idxmin()

def avgdrawdownperc_nozero(equity,basic_quantity):
    """
    calcola la media del draw down storico
    non considerando i valori nulli (nuovi massimi di equity line)
    """
    dd_perc = drawdown_perc(equity,basic_quantity)
    return round(dd_perc[dd_perc < 0].mean(),2)
    
def profit(equity):
    return round(equity[-1],2)
    
def operation_number(operations):
    return operations.count()
    
def avg_trade(operations):
    return round(operations.mean(),2)
    
def max_draw_down(equity):
    dd = drawdown(equity)
    return round(dd.min(),2)
    
def avgdrawdown_nozero(equity):
    """
    calcola la media del draw down storico
    non considerando i valori nulli (nuovi massimi di equity line)
    """
    dd = drawdown(equity)
    return round(dd[dd < 0].mean(),2)

def drawdown_statistics(operations):
    dd = drawdown(operations)
    return dd.describe(percentiles=[0.30, 0.20, 0.10, 0.05, 0.01])

def avg_loss(operations):
    return round(operations[operations < 0].mean(),2)
    
def max_loss(operations):
    return round(operations.min(),2)
    
def max_loss_date(operations):
    return operations.idxmin()
    
def avg_gain(operations):
    return round(operations[operations > 0].mean(),2)
    
def max_gain(operations):
    return round(operations.max(),2)
    
def max_gain_date(operations):
    return operations.idxmax()
    
def gross_profit(operations):
    return round(operations[operations > 0].sum(),2)
    
def gross_loss(operations):
    return round(operations[operations <= 0].sum(),2)
    
def profit_factor(operations):
    a = gross_profit(operations)
    b = gross_loss(operations)
    if b != 0:
        return round(abs(a / b), 2)
    else:
        return round(abs(a / 0.00000001), 2)
        
def percent_win(operations):
    return round((operations[operations > 0].count() / operations.count() * 100),2)
    
def reward_risk_ratio(operations):
    if operations[operations <= 0].mean() != 0:
        return round((operations[operations > 0].mean() / -operations[operations <= 0].mean()),2)
    else:
        return np.inf
    
def sustainability(operations):
    pw = percent_win(operations)
    rrr = reward_risk_ratio(operations)
    return pw * rrr - (100 - pw) * (1 / rrr)
        
def delay_between_peaks(equity):
    """
    Funzione per calcolare i ritardi istantanei in barre
    nel conseguire nuovi massimi di equity line
    Input: equity line
    """
    work_df = pd.DataFrame(equity, index = equity.index)
    work_df["drawdown"] = drawdown(equity)
    work_df["delay_elements"] = work_df["drawdown"].apply(lambda x: 1 if x < 0 else 0)
    work_df["resets"] = np.where(work_df["drawdown"] == 0, 1, 0)
    work_df['cumsum'] = work_df['resets'].cumsum()
    #print(work_df.iloc[-20:,:])
    a = pd.Series(work_df['delay_elements'].groupby(work_df['cumsum']).cumsum())
    return a

def max_delay_between_peaks(equity):
    """
    Funzione per calcolare il più lungo ritardo in barre dall'ultimo massimo
    Input: equity line
    """
    a = delay_between_peaks(equity)
    return a.max()
    
def avg_delay_between_peaks(equity):
    """
    Funzione per calcolare il ritardo medio in barre
    nel conseguire nuovi massimi di equity line
    Input: equity line
    """
    work_df = pd.DataFrame(equity, index = equity.index)
    work_df["drawdown"] = drawdown(equity)
    work_df["delay_elements"] = work_df["drawdown"].apply(lambda x: 1 if x < 0 else np.nan)
    work_df["resets"] = np.where(work_df["drawdown"] == 0, 1, 0)
    work_df['cumsum'] = work_df['resets'].cumsum()
    work_df.dropna(inplace = True)
    a = work_df['delay_elements'].groupby(work_df['cumsum']).sum()
    return round(a.mean(),2)
    
def old_omega_ratio(operations,threshold):
    downside=0
    upside=0
    i=0
    while i < len(operations):
        if operations[i] < threshold:
            downside += (threshold - operations[i])
        if operations[i] > threshold:
            upside += (operations[i] - threshold)
        i+=1
    if downside != 0:
        return round(upside / downside,2)
    else:
        return np.inf
    
def omega_ratio(operations,threshold):
    upside = np.where(operations > threshold, operations - threshold, 0).sum()
    downside = np.where(operations < threshold, threshold - operations, 0).sum()
    
    if downside != 0:
        return round(upside / downside, 2)
    else:
        return np.inf
    
def old_sharpe_ratio(operations):
    """
    Il rapporto tra il guadagno totale 
    e la deviazione standard dell'equity line
    """
    equity = operations.cumsum()
    netprofit = equity[-1]
    std = equity.std()
    if std != 0:
        return round(netprofit / std,2)
    else:
        return np.inf

def sharpe_ratio_yearly(operations):
    """
    Il rapporto medio su base annuale tra
    il guadagno annuale ed il draw down annuale
    """
    yearly_operations = operations.resample('A').sum()
    yearly_std = operations.resample('A').std()
    records = []
    
    for i in range(len(yearly_operations)):
        if yearly_std[i] != 0:
            records.append(yearly_operations[i] / yearly_std[i])
        else:
            records.append(np.inf)
            
    records = pd.Series(records, index = yearly_operations.index)
    
    return round(records.mean(),2), records

def sharpe_ratio_old(operations, start_nav, period_risk_free):
    """
    Sharpe Ratio
    Rocket Capital Investment Version 2021
    
    operations: list of trades
    start_nav: starting money
    period_risk_free: annual risk free percent profit
    """
    results = pd.DataFrame(operations.resample('A').sum())
    results.columns = ["yearly_profit"]
    results["yearly_returns"] = results.yearly_profit / start_nav * 100
    results["excess_return"] = results.yearly_returns - period_risk_free
    results["yearly_std"] = operations.resample('A').std()
    
    avg_excess_return = results.excess_return.mean()
    risk = results.yearly_std.mean()
    if risk != 0:
        sharpe = round(avg_excess_return / risk * 100,2)
    else:
        if avg_excess_return >= 0:
            sharpe = np.inf
        else:
            sharpe = -np.inf
    return sharpe

def sharpe_ratio_high_value(operations, start_nav, period_risk_free):
    """
    Sharpe Ratio
    Rocket Capital Investment Version Version 20211023
    
    operations: list of trades
    start_nav: starting money
    period_risk_free: annual risk free percent profit
    """
    results = pd.DataFrame(operations.resample('A').sum())
    results.columns = ["yearly_profit"]
    results["yearly_returns"] = results.yearly_profit / start_nav * 100
    results["excess_return"] = results.yearly_returns - period_risk_free
    results["yearly_std"] = operations.resample('A').std()
    results["yearly_std_perc"] = results.yearly_std / results.yearly_profit * 100
    
    avg_excess_return = results.excess_return.mean()
    risk = results.yearly_std_perc.mean()
    #print(results)
    #print(avg_excess_return, risk)
    if risk != 0:
        sharpe = round(avg_excess_return / risk,2)
    else:
        if avg_excess_return >= 0:
            sharpe = np.inf
        else:
            sharpe = -np.inf
    return sharpe

def sharpe_ratio(operations, start_nav, period_risk_free):
    """
    Sharpe Ratio
    Rocket Capital Investment Version 20211023
    
    operations: list of trades
    start_nav: starting money
    period_risk_free: annual risk free percent profit
    """
    results = pd.DataFrame(operations.resample('A').sum())
    results.columns = ["yearly_profit"]
    results["yearly_returns"] = results.yearly_profit / start_nav * 100
    results["excess_return"] = results.yearly_returns - period_risk_free
    
    avg_excess_return = results.excess_return.mean()
    risk = results.yearly_returns.std()
    #print(results)
    #print(avg_excess_return, risk)
    if risk != 0:
        sharpe = round(avg_excess_return / risk,2)
        #sharpe = round(avg_excess_return / risk,2)
    else:
        if avg_excess_return >= 0:
            sharpe = np.inf
        else:
            sharpe = -np.inf
    return sharpe

def equity_daily_returns(equity,initial_quantity):
    
    real_equity = equity + initial_quantity
    df =pd.DataFrame(real_equity)
    returns = []

    for i in range(len(real_equity)):
        if i > 0:
            el = (real_equity[i]/real_equity[i-1] - 1.0) 

            if real_equity[i] >= 0 and real_equity[i-1] >= 0:
                returns.append(el)
            elif real_equity[i] < 0:
                returns.append(-el)
            else:
                returns.append(el)
        else:
            returns.append(np.nan)
                  
    df["returns"] = returns
    #df.returns.dropna()

    return df.returns#[dr!=np.inf]

def sharpe_ratio_perc(equity, start_nav, period_risk_free):
    """
    Sharpe Ratio based on daily returns
    Rocket Capital Investment Version 20220312
    
    equity: open equity
    start_nav: starting money
    period_risk_free: annual risk free percent profit
    """
    
    returns = equity_daily_returns(equity,start_nav)
    
    results = pd.DataFrame(returns - period_risk_free)
    results.columns = ["excess_return"]
    
    avg_excess_return = results.excess_return.mean()
    risk = results.excess_return.std()
    sharpe = avg_excess_return / risk * np.sqrt(252)
    return sharpe

def sortino_ratio(operations, start_nav, monthly_risk_free):
    """
    Sortino Ratio
    Gandalf Project Version 2020
    
    operations: list of trades
    start_nav: starting money
    monthly_risk_free: monthly risk free percent profit
    """
    results = pd.DataFrame(operations.resample('M').sum())
    results.columns = ["monthly_operations"]
    results["monthly_returns"] = results.monthly_operations / start_nav * 100
    
    monthly_avg = []
    for i in range(1,13):
        monthly_avg.append(results[results.monthly_returns.index.month == i].monthly_returns.mean())
        
    grid = pd.DataFrame(monthly_avg, index = range(1,13), columns = ["avg_returns"])
    grid["excess_return"] = grid.avg_returns - monthly_risk_free
    grid["negative_excees_return"] = np.where(grid.excess_return < 0, grid.excess_return, 0)
    
    avg_excess_return = grid.excess_return.mean()
    downside_risk = grid.negative_excees_return.apply(lambda x: x ** 2).sum() ** 0.5
    
    return round(avg_excess_return / downside_risk,2)

def kestner_ratio(operations):
    """
    Kestner Ratio versione 2003
    Una volta calcolata l'equity line dei contributi mensili delle operazioni aggregate
    calcoliamo la retta di regressione che meglio approssima lo sciame di punti 
    e riportiamo il rapporto tra la pendenza di tale retta e l'errore standard
    tra ogni punto e la retta medesima
    
    operations: list of trades
    """
    import numpy as np
    import matplotlib.pyplot as plt 
    from scipy import stats
    
    monthly_operations = operations.resample('M').sum().fillna(0)
    monthly_equity = monthly_operations.cumsum()
    index = np.array(np.arange(1,monthly_operations.count() + 1))
    
    x = index
    y = monthly_equity
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    
    if std_err != 0 and len(index) > 0:
        return round(gradient / (std_err * len(index)),2)
    else:
        return np.inf
    
def old_calmar_ratio(operations):
    """
    Il rapporto tra il guadagno finale e il max draw down registrato
    """
    equity = operations.cumsum()
    netprofit = equity[-1]
    mdd = drawdown(operations).min()
    if mdd != 0:
        return round(-netprofit / mdd,2)
    else:
        return np.inf
    
def calmar_ratio_yearly(operations):
    """
    Il rapporto medio su base annuale tra
    il guadagno annuale ed il draw down annuale
    """
    yearly_operations = operations.resample('A').sum()
    yearly_drawdown = drawdown(operations).resample('A').min()
    records = []
    
    for i in range(len(yearly_operations)):
        if yearly_drawdown[i] != 0:
            records.append(-yearly_operations[i] / yearly_drawdown[i])
        else:
            records.append(np.inf)
            
    records = pd.Series(records, index = yearly_operations.index)
    return round(records.mean(),2), records
    
def calmar_ratio_zeros(operations):
    """
    Calmar Ratio
    Gandalf Project Version 2020
    
    operations: list of trades
    """
    global_equity = operations.cumsum()

    results = pd.DataFrame(operations.resample('A').sum())
    results.columns = ["yearly_profits"]
    
    group_years = results.groupby(by = results.index.year).sum()
    years = list(group_years.index)
    
    equities = []
    drawdowns = []
    yearly_calmar = []
    for year in years:
        equities.append(operations[operations.index.year == year].cumsum())
        drawdowns.append(drawdown(equities[-1]).min())
            
    results["yearly_drawdowns"] = drawdowns
    results["yearly_calmars"] = results.yearly_profits / -results.yearly_drawdowns
    
    return round(results["yearly_calmars"].mean(),2)

def calmar_ratio(operations):
    """
    Calmar Ratio
    Gandalf Project Version 2022
    Version to avoid cases with yearly draw downs equal to zero!
    
    operations: list of trades
    """
    global_equity = operations.cumsum()

    results = pd.DataFrame(operations.resample('A').sum())
    results.columns = ["yearly_profits"]
    
    group_years = results.groupby(by = results.index.year).sum()
    years = list(group_years.index)
    
    equities = []
    drawdowns = []
    yearly_calmar = []
    for year in years:
        equities.append(operations[operations.index.year == year].cumsum())
        dd = drawdown(equities[-1]).min()
        drawdowns.append(dd)
            
    results["yearly_drawdowns"] = drawdowns
    results["yearly_calmars"] = results.yearly_profits / abs(results.yearly_drawdowns)
    return round(results[results.yearly_calmars != np.inf].yearly_calmars.mean(),2)

def cagr(operations,initial_capital):
    """
    Compound Annual Growth Rate CAGR   
    Gandalf Project Version 2020
    
    operations: list of trades
    initial_capital: capital at the beginning of the investment
    """
    ending_capital = initial_capital + operations.sum()
    
    results = pd.DataFrame(operations.resample('A').sum())
    results.columns = ["yearly_profits"]
    
    group_years = results.groupby(by = results.index.year).sum()
    years = list(group_years.index)

    result = (ending_capital / initial_capital) ** (1 / len(years)) - 1
                 
    return round(result * 100,2)

def annual_return(operations,initial_capital):
    """
    Yearly Return  
    Gandalf Project Version 2020
    
    operations: list of trades
    initial_capital: capital at the beginning of the investment
    """
    ending_capital = initial_capital + operations.sum()
    
    results = pd.DataFrame(operations.resample('A').sum())
    results.columns = ["yearly_profits"]
    
    group_years = results.groupby(by = results.index.year).sum()
    years = list(group_years.index)

    result = (ending_capital - initial_capital) / initial_capital * 100 / len(years)
           
    return round(result,2)

# PERFORMANCE METRICS *******************************************************************************************************end

# PLOTTING FUNCTIONS ******************************************************************************************************start

def plot_equity(equity,color):
    """
    Funzione per stampare un'equity line
    """
    new_highs = equity.expanding().max()
    limes = pd.DataFrame(np.where(equity == new_highs, new_highs, np.nan), index = equity.index)
    
    plt.figure(figsize = (14, 8), dpi=300)
    plt.plot(equity, color = color)
    plt.plot(limes, color = "lime", marker =".", markersize = 6)
    plt.xlabel("Time")
    plt.ylabel("Profit/Loss")
    plt.title('Equity Line - by Gandalf Project R&D')
    plt.xticks(rotation='vertical')
    plt.grid(True)
    plt.show()
    return
    
def plot_double_equity(closed_equity,open_equity):
    """
    Funzione per stampare due equity sovrapposte
    """
    plt.figure(figsize=(14, 8), dpi=300)
    plt.plot(open_equity, color='red')
    plt.plot(closed_equity, color='green')
    plt.xlabel("Time")
    plt.ylabel("Profit/Loss")
    plt.title('Open & Closed Equity Line - by Gandalf Project R&D')
    plt.xticks(rotation='vertical')
    plt.grid(True)
    plt.show()
    return

def plot_equities(equity1, equity2):
    """
    Funzione per stampare un'equity line
    """
    plt.figure(figsize=(14, 8), dpi=300)
    plt.plot(equity1, color="green")
    plt.plot(equity2, color="red")
    plt.plot(equity1 + equity2, color="blue")
    plt.xlabel("Time")
    plt.ylabel("Profit/Loss")
    plt.title('Equity Lines - by Gandalf Project R&D')
    plt.xticks(rotation='vertical')
    plt.grid(True)
    plt.show()
    return

def plot_equity_price(data, equity):
    
    from pandas.plotting import register_matplotlib_converters
    
    new_highs = equity.expanding().max()
    limes = pd.DataFrame(np.where(equity == new_highs, new_highs, np.nan), index = equity.index)
    
    fig, ax1 = plt.subplots(figsize = [14, 8], dpi = 300)
    
    ax1.set_title("Equity vs Underlying - Powered by Gandalf Project")
    ax1.plot(data.close, color = 'tan', lw = 2, alpha = 0.5)
    #ax1.legend(["Asset"])
    #ax1.fill_between(dataset.index, 0, dataset.close, alpha = 0.3)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Asset Price")
    ax1.grid()
    
    ax2 = ax1.twinx()
    ax2.plot(equity, color='green')
    ax2.plot(limes, color = "lime", marker =".", markersize = 6)
    ax2.set_ylabel("Strategy Profit/Loss")
    ax2.legend(["Equity Line"])

    plt.show()
    return
    
def plot_drawdown(equity,color):
    """
    Funzione per graficare la curva di draw down
    """
    dd = drawdown(equity)
    plt.figure(figsize = (12, 6), dpi = 300)
    plt.plot(dd, color = color)
    plt.fill_between(dd.index, 0, dd, color = color)
    plt.xlabel("Time")
    plt.ylabel("Monetary Loss")
    plt.title('Draw Down - by Gandalf Project R&D')
    plt.xticks(rotation='vertical')
    plt.grid(True)
    plt.show()
    return

def plot_drawdown_perc(equity,basic_quantity,color):
    """
    Funzione per graficare la curva di draw down percentuale
    """
    real_equity = basic_quantity + equity
    eq_max = real_equity.expanding().max()
    dd = drawdown(equity)
    dd_perc = (dd / eq_max) * 100
    dd_perc = pd.Series(np.where(dd_perc > 0, 0 , dd_perc), index = equity.index)
    plt.figure(figsize = (12, 6), dpi = 300)
    plt.plot(dd_perc, color = color)
    plt.fill_between(dd_perc.index, 0, dd_perc, color = color)
    plt.xlabel("Time")
    plt.ylabel("Percentage Loss")
    plt.title('Percentage Draw Down - by Gandalf Project R&D')
    plt.xticks(rotation='vertical')
    plt.grid(True)
    plt.show()
    return

def plot_drawdown_perc_interactive(equity,basic_quantity):
    
    real_equity = basic_quantity + equity
    eq_max = real_equity.expanding().max()
    dd = drawdown(equity)
    dd_perc = (dd / eq_max) * 100
    dd_perc = pd.DataFrame(np.where(dd_perc > 0, 0 , dd_perc), index = equity.index)
    dd_perc.columns = ["drawdown"]    
    #dd = pd.DataFrame(drawdown(equity), index = equity.index)
    #dd_perc.columns = ["drawdown"]
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = dd_perc.index, y = dd_perc.drawdown, marker_color = "red", 
                             fill='tozeroy', mode='none', fillcolor = "red", name = 'Draw Down'))
    
    fig.update_traces(opacity = 0.75)
    
    fig.update_xaxes(showgrid = True, zeroline = True, gridwidth = 1, gridcolor = 'lightgrey')
    fig.update_yaxes(showgrid = True, zeroline = True, gridwidth = 1, gridcolor = 'lightgrey')

    fig.update_layout(
    title_text = 'Draw Down Percentage - by Gandalf Project R&D', # title of plot
    xaxis_title_text = 'Time', # xaxis label
    yaxis_title_text = 'Draw Down', # yaxis label
    bargap = 0.1, # gap between bars of adjacent location coordinates
    bargroupgap = 0.1, # gap between bars of the same location coordinates
    plot_bgcolor = 'white')
    fig.show()
    return
    
def plot_equity_interactive_old(equity, color):
    equity.iplot(kind = 'line', color = color, title="Equity Line - by Gandalf Project R&D")
    plt.show()
    return

def plot_equity_interactive(equity, color):

    import plotly.graph_objects as go
    
    new_highs = equity.expanding().max()
    limes = np.where(equity == new_highs, new_highs, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = equity.index, y = equity, marker_color = color, 
                             mode = 'lines', name = 'equity'))
    fig.add_trace(go.Scatter(x = equity.index, y = limes, marker_color = "lime",
                             mode = 'markers', name = 'new highs'))
    
    fig.update_xaxes(showgrid = True, zeroline = True, gridwidth = 1, gridcolor = 'lightgrey')
    fig.update_yaxes(showgrid = True, zeroline = True, gridwidth = 1, gridcolor = 'lightgrey')

    fig.update_layout(
    title_text = 'Equity Line - by Gandalf Project R&D', # title of plot
    xaxis_title_text = 'Time', # xaxis label
    yaxis_title_text = 'Profit/Loss', # yaxis label
    bargap = 0.1, # gap between bars of adjacent location coordinates
    bargroupgap = 0.1, # gap between bars of the same location coordinates
    plot_bgcolor = 'white')

    fig.show()
    return

def plot_drawdown_interactive_old(operations):
    dd = pd.DataFrame(drawdown(operations), index=operations.index)
    dd.iplot(kind='line', color="red", title="Draw Down Histogram - by Gandalf Project R&D")
    plt.show()
    return

def plot_drawdown_interactive(equity):
    
    dd = pd.DataFrame(drawdown(equity), index = equity.index)
    dd.columns = ["drawdown"]
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = dd.index, y = dd.drawdown, marker_color = "red", 
                             fill='tozeroy', mode='none', fillcolor = "red", name = 'Draw Down'))
    
    fig.update_traces(opacity = 0.75)
    
    fig.update_xaxes(showgrid = True, zeroline = True, gridwidth = 1, gridcolor = 'lightgrey')
    fig.update_yaxes(showgrid = True, zeroline = True, gridwidth = 1, gridcolor = 'lightgrey')

    fig.update_layout(
    title_text = 'Draw Down Histogram - by Gandalf Project R&D', # title of plot
    xaxis_title_text = 'Time', # xaxis label
    yaxis_title_text = 'Draw Down', # yaxis label
    bargap = 0.1, # gap between bars of adjacent location coordinates
    bargroupgap = 0.1, # gap between bars of the same location coordinates
    plot_bgcolor = 'white')

    fig.show()
    return

def plot_monthly_histogram(operations):
    monthly = operations.resample('M').sum()
    colors = pd.Series()
    colors = monthly.apply(lambda x: "green" if x > 0 else "red")
    n_groups = len(monthly)
    plt.subplots(figsize=(14, 8), dpi=300)
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 1

    rects1 = plt.bar(index,
                     monthly,
                     bar_width,
                     alpha=opacity,
                     color=colors,
                     label='Monthly Profit-Loss')

    plt.xlabel('Months')
    plt.ylabel('Profit - Loss')
    plt.title('Monthly Profit-Loss - by Gandalf Project R&D')
    # plt.xticks(monthly.count(), monthly.index.month, rotation=90)
    # plt.legend()
    plt.grid(False)
    plt.show()
    
def plot_monthly_histogram_interactive(operations):
    monthly = operations.resample('M').sum()
    monthly = monthly.fillna(0)
    monthly_positive = monthly[monthly >= 0]
    monthly_negative = monthly[monthly < 0]
    df = pd.DataFrame({"Positive": monthly_positive, "Negative": monthly_negative}).fillna(0)
    df.iplot(kind='bar', color=["red","green"], title="Monthly Profit-Loss - by Gandalf Project R&D")
    return
    
def plot_annual_histogram(operations):
    yearly = operations.resample('A').sum()
    colors = pd.Series(dtype="float64")
    colors = yearly.apply(lambda x: "green" if x > 0 else "red")
    n_groups = len(yearly)
    plt.subplots(figsize=(14, 8), dpi = 300)
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 1

    rects1 = plt.bar(index,
                     yearly,
                     bar_width,
                     alpha=opacity,
                     color=colors,
                     label='Yearly Statistics')

    plt.xlabel('Years')
    plt.ylabel('Profit - Loss')
    plt.title('Yearly Profit-Loss - by Gandalf Project R&D')
    plt.xticks(index, yearly.index.year, rotation=90)
    plt.grid(True)
    plt.show()
    return

def plot_annual_histogram_interactive_old(operations):
    yearly = operations.resample('A').sum()
    yearly = yearly.fillna(0)
    yearly_positive = yearly[yearly >= 0]
    yearly_negative = yearly[yearly < 0]
    df = pd.DataFrame({"Positive": yearly_positive, "Negative": yearly_negative}).fillna(0)
    df.iplot(kind='bar', color=["green","red"], title="Yearly Profit-Loss - by Gandalf Project R&D")
    return

def plot_annual_histogram_interactive(operations):
    yearly = operations.resample('A').sum()
    yearly = yearly.fillna(0)
    yearly_positive = yearly[yearly >= 0]
    yearly_negative = yearly[yearly < 0]
    df = pd.DataFrame({"Positive": yearly_positive, "Negative": yearly_negative}).fillna(0)
    #df.iplot(kind='bar', color=["green","red"], title="Yearly Profit-Loss - by Gandalf Project R&D")
    print(df)
    import plotly.graph_objects as go
    
    fig = go.Figure(data = [go.Bar(x = df.index.year, y = df.Positive, marker_color = "green", name = "Positive")])
    fig.add_trace(go.Bar(x = df.index.year, y = df.Negative, marker_color = "red", name = "Negative"))
    
    fig.update_traces(marker_line_color='black',
                      marker_line_width=1.5, opacity=0.6)

    fig.update_layout(
    title_text = 'Yearly Profit-Loss - by Gandalf Project R&D', # title of plot
    xaxis_title_text = 'Years', # xaxis label
    yaxis_title_text = 'Profit/Loss', # yaxis label
    bargap = 0.1, # gap between bars of adjacent location coordinates
    bargroupgap = 0.1 # gap between bars of the same location coordinates
    )

    fig.update_xaxes(showgrid = True, zeroline = True, gridwidth = 1, gridcolor = 'lightgrey')
    fig.update_yaxes(showgrid = True, zeroline = True, gridwidth = 1, gridcolor = 'lightgrey')

    fig.update_traces(opacity = 0.75)
    fig['layout'].update(plot_bgcolor = 'white')
    fig.update_layout(barmode='stack', xaxis_tickangle=-45)

    fig.update_xaxes(side = "bottom")
    fig.show()
    return
    
def plot_monthly_bias_histogram(operations):
    monthly = pd.DataFrame(operations.fillna(0)).resample('M').sum()
    monthly['Month'] = monthly.index.month
    biasMonthly = []
    months = []

    for month in range(1, 13):
        months.append(month)
    for month in months:
        biasMonthly.append(monthly[(monthly['Month'] == month)].mean())

    biasMonthly = pd.DataFrame(biasMonthly)
    column = biasMonthly.columns[0]
    colors = pd.Series(dtype="float64")
    colors = biasMonthly[column].apply(lambda x: "green" if x > 0 else "red")
    n_groups = len(biasMonthly)
    plt.subplots(figsize=(14, 6), dpi=300)
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 1

    rects1 = plt.bar(index,
                     biasMonthly[column],
                     bar_width,
                     alpha=opacity,
                     color=colors,
                     label='Yearly Statistics')

    plt.xlabel('Months')
    plt.ylabel('Average Profit - Loss')
    plt.title('Average Monthly Profit-Loss - by Gandalf Project R&D')
    months_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                    "October", "November", "December"]
    plt.xticks(index, months_names, rotation=45)
    plt.grid(True)
    plt.show()
    return

def plot_monthly_bias_histogram_interactive_old(operations):
    monthly = pd.DataFrame(operations.fillna(0)).resample('M').sum()
    monthly['Month'] = monthly.index.month
    biasMonthly = []
    months = []

    for month in range(1, 13):
        months.append(month)

    for month in months:
        biasMonthly.append(monthly[(monthly['Month'] == month)].mean())

    biasMonthly = pd.DataFrame(biasMonthly).fillna(0)
    biasMonthly.index = biasMonthly.Month
    biasMonthly.drop("Month", axis=1, inplace=True)
    biasMonthly_positive = biasMonthly[biasMonthly >= 0]
    biasMonthly_negative = biasMonthly[biasMonthly < 0]
    biasMonthly_positive_series = biasMonthly_positive.operations
    biasMonthly_negative_series = biasMonthly_negative.operations
    df = pd.DataFrame({"Positive": biasMonthly_positive_series, "Negative": biasMonthly_negative_series}).fillna(0)
    df.iplot(kind='bar', color=["green","red"], title="Average Monthly Profit-Loss - by Gandalf Project R&D")
    return

def plot_monthly_bias_histogram_interactive(operations):
    monthly = pd.DataFrame(operations.fillna(0)).resample('M').sum()
    monthly['Month'] = monthly.index.month
    biasMonthly = []
    months = []

    for month in range(1, 13):
        months.append(month)

    for month in months:
        biasMonthly.append(monthly[(monthly['Month'] == month)].mean())

    biasMonthly = pd.DataFrame(biasMonthly).fillna(0)
    biasMonthly.index = biasMonthly.Month
    biasMonthly.drop("Month", axis=1, inplace=True)
    biasMonthly_positive = biasMonthly[biasMonthly >= 0]
    biasMonthly_negative = biasMonthly[biasMonthly < 0]
    biasMonthly_positive_series = biasMonthly_positive.operations.apply(lambda x: round(x,2))
    biasMonthly_negative_series = biasMonthly_negative.operations.apply(lambda x: round(x,2))
    df = pd.DataFrame({"Positive": biasMonthly_positive_series, "Negative": biasMonthly_negative_series}).fillna(0)
    #df.iplot(kind='bar', color=["green","red"], title="Average Monthly Profit-Loss - by Gandalf Project R&D")

    month_names = ["January","February", "March", "April", "May", "June", "July", "August",
                   "September", "October", "November", "December"]
    
    df.index = month_names
    
    print(df)
    import plotly.graph_objects as go
    
    fig = go.Figure(data = [go.Bar(x = df.index,
                                   y = df.Positive, marker_color = "green", name = "Positive")])
    fig.add_trace(go.Bar(x = df.index, y = df.Negative, marker_color = "red", name = "Negative"))
    
    fig.update_traces(marker_line_color='black',
                      marker_line_width=1.5, opacity=0.6)

    fig.update_layout(
    title_text = 'Average Monthly Profit-Loss - by Gandalf Project R&D', # title of plot
    xaxis_title_text = 'Months', # xaxis label
    yaxis_title_text = 'Average Profit/Loss', # yaxis label
    bargap = 0.1, # gap between bars of adjacent location coordinates
    bargroupgap = 0.1 # gap between bars of the same location coordinates
    )
    
    fig.update_xaxes(showgrid = True, zeroline = True, gridwidth = 1, gridcolor = 'lightgrey')
    fig.update_yaxes(showgrid = True, zeroline = True, gridwidth = 1, gridcolor = 'lightgrey')

    fig.update_traces(opacity = 0.75)
    fig['layout'].update(plot_bgcolor = 'white')
    fig.update_layout(barmode='stack', xaxis_tickangle=-45)

    fig.update_xaxes(side = "bottom")
    fig.show()
    
    return
    
def plot_equity_heatmap(operations,annotations):
    monthly = operations.resample('M').sum()
    toHeatMap = pd.DataFrame(monthly)
    toHeatMap["Year"] = toHeatMap.index.year
    toHeatMap["Month"] = toHeatMap.index.month
    Show = toHeatMap.groupby(by=['Year','Month']).sum().unstack()
    #if len(Show) > 1:
     #   Show.columns = ["January","February","March","April","May","June",
      #                  "July","August","September","October","November","December"]
    #else:
     #   print("WARNING from line 525 in plot_equity_heatmap function: not enought months to plot! ", len(Show))
      #  return
    try:
        Show.columns = ["January","February","March","April","May","June",
                        "July","August","September","October","November","December"]
    except:
        print("WARNING from line 525 in plot_equity_heatmap function: not enought months to plot! ", len(Show))
        return
    plt.figure(figsize=(8,6),dpi=120)
    sns.heatmap(Show, cmap="RdYlGn", linecolor="white", linewidth=0.1, annot=annotations, 
                vmin=-max(monthly.min(), monthly.max()), vmax=monthly.max())
    return

def plot_equity_heatmap_interactive(operations,annotations):
    monthly = operations.resample('M').sum()
    toHeatMap = pd.DataFrame(monthly)
    toHeatMap["Year"] = toHeatMap.index.year
    toHeatMap["Month"] = toHeatMap.index.month
    Show = toHeatMap.groupby(by=['Year','Month']).sum().unstack()
    #if len(Show) > 1:
     #   Show.columns = ["January","February","March","April","May","June",
      #                  "July","August","September","October","November","December"]
    #else:
     #   print("WARNING from line 525 in plot_equity_heatmap function: not enought months to plot! ", len(Show))
      #  return
    try:
        Show.columns = ["January","February","March","April","May","June",
                        "July","August","September","October","November","December"]
    except:
        print("WARNING from line 525 in plot_equity_heatmap function: not enought months to plot! ", len(Show))
        return

    import plotly.graph_objects as go
    
    layout = go.Layout(xaxis = go.layout.XAxis(title=go.layout.xaxis.Title(text='Months',)),
                       yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(text='Years',)))
    
    fig = go.Figure(data = go.Heatmap(z = Show, x = Show.columns, y = Show.index,
                                      type = 'heatmap', hoverongaps = False, colorscale = "RdYlGn",
                                      zmin = -max(monthly.min(), monthly.max()), zmax = monthly.max()), 
                    layout=layout)
    
    fig.update_layout(margin = dict(t = 50,r = 10, b = 10, l = 10),
                      width = 750, height = 650,
                      showlegend = True, autosize = True)

    fig['layout']['yaxis']['autorange'] = "reversed"
    fig['layout'].update(plot_bgcolor = 'white')
    fig.update_xaxes(side = "bottom")
    fig.show()
    
    return

def plot_mae(tradelist):
    pos_tradelist = tradelist[tradelist.operations > 0]
    neg_tradelist = tradelist[tradelist.operations <= 0]

    plt.figure(figsize = [12,8], dpi= 300)
    plt.scatter(-pos_tradelist.mae, pos_tradelist.operations.apply(lambda x: x if x>=0 else -x), 
                marker = "^", color = "green")
    plt.scatter(-neg_tradelist.mae, neg_tradelist.operations.apply(lambda x: x if x>=0 else -x), 
                marker = "v", color = "red")
    plt.title("Maximum Adverse Excursion - by Gandalf Project R&D")
    plt.xlabel("Maximum Trade DrawDown")
    plt.ylabel("Trade Profit/Loss")

    plt.grid()
    plt.show()
    
def plot_mfe(tradelist):
    pos_tradelist = tradelist[tradelist.operations > 0]
    neg_tradelist = tradelist[tradelist.operations <= 0]

    plt.figure(figsize = [12,8], dpi= 300)
    plt.scatter(pos_tradelist.mfe, pos_tradelist.operations.apply(lambda x: x if x>=0 else -x), 
                marker = "^", color = "green")
    plt.scatter(neg_tradelist.mfe, neg_tradelist.operations.apply(lambda x: x if x>=0 else -x), 
                marker = "v", color = "red")
    plt.title("Maximum Favorable Excursion - by Gandalf Project R&D")
    plt.xlabel("Maximum Trade RunUp")
    plt.ylabel("Trade Profit/Loss")

    plt.grid()
    plt.show()
    
def plot_trade_duration(tradelist):
    pos_tradelist = tradelist[tradelist.operations > 0]
    neg_tradelist = tradelist[tradelist.operations <= 0]

    plt.figure(figsize = [12,8], dpi= 300)
    plt.scatter(pos_tradelist.bars_in_trade, pos_tradelist.operations.apply(lambda x: x if x>=0 else -x), 
                marker = "^", color = "green")
    plt.scatter(neg_tradelist.bars_in_trade, neg_tradelist.operations.apply(lambda x: x if x>=0 else -x), 
                marker = "v", color = "red")
    plt.title("Time in Trade - by Gandalf Project R&D")
    plt.xlabel("Trade Duration in Bars")
    plt.ylabel("Trade Profit/Loss")

    plt.grid()
    plt.show()
    
def plot_capital(tl,margin_percent):
    """
    Funzione per graficare il controvalore associato a ciascun trade
    """
    plt.figure(figsize = (12, 6), dpi = 300)
    plt.plot(tl.capital, color = "green")
    plt.fill_between(tl.index, 0, tl.capital, color = "green")
    plt.plot(tl.capital * margin_percent / 100, color = "lime")
    plt.fill_between(tl.index, 0, tl.capital * margin_percent / 100, color = "lime")
    plt.xlabel("Date")
    plt.ylabel("Capital for each trade")
    plt.title('Capital - by Gandalf Project R&D')
    plt.xticks(rotation = 'vertical')
    plt.grid(True)
    plt.show()
    return

def plot_capital_interactive(tl):
       
    #dd = pd.DataFrame(drawdown(equity), index = equity.index)
    #dd_perc.columns = ["drawdown"]
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = tl.index, y = tl.capital, marker_color = "green", 
                             fill='tozeroy', mode='none', fillcolor = "green", name = 'Capital'))
    
    fig.update_traces(opacity = 0.75)
    
    fig.update_xaxes(showgrid = True, zeroline = True, gridwidth = 1, gridcolor = 'lightgrey')
    fig.update_yaxes(showgrid = True, zeroline = True, gridwidth = 1, gridcolor = 'lightgrey')

    fig.update_layout(
    title_text = 'Capital - by Gandalf Project R&D', # title of plot
    xaxis_title_text = 'Date', # xaxis label
    yaxis_title_text = 'Capital for each trade', # yaxis label
    bargap = 0.1, # gap between bars of adjacent location coordinates
    bargroupgap = 0.1, # gap between bars of the same location coordinates
    plot_bgcolor = 'white')
    fig.show()
    return
    
def performance_report(dataset,
                       tradelist,
                       closed_equity,
                       open_equity,
                       initial_capital,
                       risk_free_rate,
                       margin_percent,
                       interactive):
    
    print(Fore.LIGHTGREEN_EX + "*****************************************************************************************")
    print("*** Performance Report - by Gandalf Project R&D - Version Wintermute - Copyright 2022 ***")
    print("*****************************************************************************************" + Fore.RESET)
    if tradelist.empty:
        print("")
        print("Nessuna operazione registrata!")
        return
    else: 
        print("")
        
        my_CAGR = cagr(tradelist.operations,initial_capital)
        if my_CAGR > 0:
            print("CAGR:                     " + Fore.LIGHTGREEN_EX + str(my_CAGR) + Fore.LIGHTWHITE_EX + " (capital = " + str(initial_capital) + ")" + Fore.RESET)
        else:
            print("CAGR:                     " + Fore.RED + str(my_CAGR) + Fore.LIGHTWHITE_EX + " (capital = " + str(initial_capital) + ")" + Fore.RESET)
        
        my_annual_return = annual_return(tradelist.operations,initial_capital)
        if my_annual_return > 0:
            print("Annual Return:            " + Fore.LIGHTGREEN_EX + str(my_annual_return) + Fore.LIGHTWHITE_EX + " (capital = " + str(initial_capital) + ")" + Fore.RESET)
        else:
            print("Annual Return:            " + Fore.RED + str(my_annual_return) + Fore.LIGHTWHITE_EX + " (capital = " + str(initial_capital) + ")" + Fore.RESET)
            
        print("")
        my_calmar_ratio = calmar_ratio(tradelist.operations)
        if my_calmar_ratio > 1:
            print("Calmar Ratio:             " + Fore.LIGHTGREEN_EX + str(my_calmar_ratio) + Fore.LIGHTWHITE_EX + " (yearly)" + Fore.RESET)
        else:
            print("Calmar Ratio:             " + Fore.RED + str(my_calmar_ratio) + Fore.LIGHTWHITE_EX + " (yearly)" + Fore.RESET)            
        my_sharpe_ratio = sharpe_ratio(tradelist.operations, initial_capital, risk_free_rate)
        if my_sharpe_ratio > 1:
            print("Sharpe Ratio:             " + Fore.LIGHTGREEN_EX + str(my_sharpe_ratio) + Fore.LIGHTWHITE_EX + " (initial capital = " + str(initial_capital) + ", risk free rate = " + str(float(risk_free_rate)) + ")" + Fore.RESET)
        else:
            print("Sharpe Ratio:             " + Fore.RED + str(my_sharpe_ratio) + Fore.LIGHTWHITE_EX + " (initial capital = " + str(initial_capital) + ", risk free rate = " + str(float(risk_free_rate)) + ")" + Fore.RESET)          
            
        my_sortino_ratio = sortino_ratio(tradelist.operations, initial_capital, risk_free_rate / 12)
        if my_sortino_ratio > 1:
            print("Sortino Ratio:            " + Fore.LIGHTGREEN_EX + str(my_sortino_ratio) + Fore.LIGHTWHITE_EX + " (initial capital = " + str(initial_capital) + ", risk free rate = " + str(float(risk_free_rate) / 12) + ")" + Fore.RESET)
        else:
            print("Sortino Ratio:            " + Fore.RED + str(my_sortino_ratio) + Fore.LIGHTWHITE_EX + " (initial capital = " + str(initial_capital) + ", risk free rate = " + str(float(risk_free_rate) / 12) + ")" + Fore.RESET)
            
        my_omega_ratio = omega_ratio(tradelist.operations,100)
        if my_omega_ratio > 1:
            print("Omega Ratio:              " + Fore.LIGHTGREEN_EX + str(my_omega_ratio) + Fore.LIGHTWHITE_EX + " (threshold = 100)" + Fore.RESET) 
        else:
            print("Omega Ratio:              " + Fore.RED + str(my_omega_ratio) + Fore.LIGHTWHITE_EX + " (threshold = 100)" + Fore.RESET) 
            
        my_kestner_ratio = kestner_ratio(tradelist.operations)
        if my_kestner_ratio > 0:
            print("Kestner Ratio:            "+ Fore.LIGHTGREEN_EX + str(my_kestner_ratio) + Fore.RESET)
        else:
            print("Kestner Ratio:            "+ Fore.RED + str(my_kestner_ratio) + Fore.RESET)
        
        print("")
        print("Operations:              ", operation_number(tradelist.operations))
        print("")
        my_profit = profit(open_equity)
        if my_profit > 0:
            print("Profit:                   " + Fore.LIGHTGREEN_EX + str(my_profit) + Fore.RESET)
        else:
            print("Profit:                   " + Fore.RED + str(my_profit) + Fore.RESET)
        my_avg_trade = avg_trade(tradelist.operations)
        if my_avg_trade > 0:
            print("Average Trade:            " + Fore.LIGHTGREEN_EX + str(my_avg_trade) + Fore.RESET)
        else:
            print("Average Trade:            " + Fore.RED + str(my_avg_trade) + Fore.RESET)            
        print("")
        my_profit_factor = profit_factor(tradelist.operations)
        if my_profit_factor > 1:
            print("Profit Factor:            " + Fore.LIGHTGREEN_EX + str(my_profit_factor) + Fore.RESET)
        else:
            print("Profit Factor:            " + Fore.RED + str(my_profit_factor) + Fore.RESET)            
        print("Gross Profit:             " + Fore.LIGHTGREEN_EX + str(gross_profit(tradelist.operations)) + Fore.RESET)
        print("Gross Loss:               " + Fore.RED + str(gross_loss(tradelist.operations)) + Fore.RESET)
        print("")
        my_percent_win = percent_win(tradelist.operations)
        if my_percent_win > 50:
            print("Percent Winning Trades:   " + Fore.LIGHTGREEN_EX + str(my_percent_win) + Fore.LIGHTWHITE_EX + " (Percent Losing Trades: " + str(round(100 - my_percent_win,2)) + ")" + Fore.RESET)
        else:
            print("Percent Winning Trades:   " + Fore.RED + str(my_percent_win) + Fore.LIGHTWHITE_EX + " (Percent Losing Trades: " + str(round(100 - my_percent_win,2)) + ")" + Fore.RESET)
        #print("Percent Losing Trades:   ", round(100 - percent_win(tradelist.operations),2))
        my_reward_risk_ratio = reward_risk_ratio(tradelist.operations)
        if my_reward_risk_ratio > 1:
            print("Reward Risk Ratio:        " + Fore.LIGHTGREEN_EX + str(my_reward_risk_ratio) + Fore.RESET)
        else:
            print("Reward Risk Ratio:        " + Fore.RED + str(my_reward_risk_ratio) + Fore.RESET)
        my_sustainability = round(sustainability(tradelist.operations),2)
        if my_sustainability > 0:
            print("Sustainability:           " + Fore.LIGHTGREEN_EX + str(my_sustainability) + Fore.RESET)
        else:
            print("Sustainability:           " + Fore.RED + str(my_sustainability) + Fore.RESET)
        print("")
        print("Trading Fees:             " + Fore.RED + str(round(tradelist.costs.sum(),2)) + Fore.RESET)
        print("")
        print("Average Gain:             " + Fore.LIGHTGREEN_EX + str(avg_gain(tradelist.operations)) + Fore.RESET)
        print("Max Gain:                 " + Fore.LIGHTGREEN_EX + str(max_gain(tradelist.operations)) + Fore.LIGHTWHITE_EX + " (" + str(max_gain_date(tradelist.operations)) + ")" + Fore.RESET)
        print("Average Loss:             " + Fore.RED + str(avg_loss(tradelist.operations)) + Fore.RESET)
        print("Max Loss:                 " + Fore.RED + str(max_loss(tradelist.operations)) + Fore.LIGHTWHITE_EX + " (" + str(max_loss_date(tradelist.operations)) + ")" + Fore.RESET)
        print("")
        print("Avg Delay Between Peaks: ", avg_delay_between_peaks(open_equity))
        print("Max Delay Between Peaks: ", max_delay_between_peaks(open_equity))
        print("")
        print("Avg Time in Trade:       ", round(tradelist.bars_in_trade.mean()))
        print("Max Time in Trade:       ", tradelist.bars_in_trade.max())
        print("Min Time in Trade:       ", tradelist.bars_in_trade.min())
        print("")
        print("Trades Standard Dev:     ", round(tradelist.operations.std(),2))
        print("Equity Standard Dev:     ", round(tradelist.operations.cumsum().std(),2))
        print("")
        print("Avg Open Draw Down:       " + Fore.RED + str(avgdrawdown_nozero(open_equity)) + Fore.RESET) 
        print("Avg Open Draw Down %:     " + Fore.RED + str(avgdrawdownperc_nozero(open_equity,initial_capital)) + "%" + Fore.RESET)
        print("Max Open Draw Down:       " + Fore.RED + str(max_draw_down(open_equity)) + Fore.LIGHTWHITE_EX + " (" + str(max_drawdown_date(open_equity)) + ")" + Fore.RESET)
        print("Max Open Draw Down %:     " + Fore.RED + str(max_draw_down_perc(open_equity,initial_capital)) + "%" + Fore.LIGHTWHITE_EX + " (" + str(max_drawdown_perc_date(open_equity,initial_capital)) + ")" + Fore.RESET)
        print("")
        print("Avg Closed Draw Down:     " + Fore.RED + str(avgdrawdown_nozero(closed_equity)) + Fore.RESET)
        print("Avg Closed Draw Down %:   " + Fore.RED +
              str(avgdrawdownperc_nozero(closed_equity,initial_capital)) + "%" + Fore.RESET)
        print("Max Closed Draw Down:     " + Fore.RED + str(max_draw_down(closed_equity)) + Fore.LIGHTWHITE_EX + " (" + str(max_drawdown_date(closed_equity)) + ")" + Fore.RESET)
        print("Max Closed Draw Down %:   " + Fore.RED +
              str(max_draw_down_perc(closed_equity,initial_capital)) + "%" + Fore.LIGHTWHITE_EX + " (" + str(max_drawdown_perc_date(closed_equity,initial_capital)) + ")" + Fore.RESET)
        print("")
        print("Draw Down Statistics:  " + Fore.RED + str(drawdown_statistics(open_equity)) + Fore.RESET)
        print("")
        print("Operation Statistics: \n")

        ExitRulesNumber = tradelist[(tradelist.exit_label == "exit_rules_long") | (tradelist.exit_label == "exit_rules_short")].exit_label.count()
        if ExitRulesNumber != 0:
            print("Exit Rule Number:        ", ExitRulesNumber, "equivalent to", 
                  round(ExitRulesNumber / operation_number(tradelist.operations) * 100, 2), "%")
            
        ExitRulesLossNumber = tradelist[(tradelist.exit_label == "exit_rules_loss_long") | (tradelist.exit_label == "exit_rules_loss_short")].exit_label.count()
        if ExitRulesLossNumber != 0:
            print("Exit Rule Loss Number:   ", ExitRulesLossNumber, "equivalent to", 
                  round(ExitRulesLossNumber / operation_number(tradelist.operations) * 100, 2), "%")
            
        ExitRulesGainNumber = tradelist[(tradelist.exit_label == "exit_rules_gain_long") | (tradelist.exit_label == "exit_rules_gain_short")].exit_label.count()
        if ExitRulesGainNumber != 0:
            print("Exit Rule Gain Number:   ", ExitRulesGainNumber, "equivalent to", 
                  round(ExitRulesGainNumber / operation_number(tradelist.operations) * 100, 2), "%")
        
        TimeExitNumbers = tradelist[(tradelist.exit_label == "time_exit_rules_long") | (tradelist.exit_label == "time_exit_rules_short")].exit_label.count()  
        if TimeExitNumbers != 0:
            print("Time Exit Number:        ", TimeExitNumbers, "equivalent to",   
                  round(TimeExitNumbers / operation_number(tradelist.operations) * 100, 2), "%")

        TimeExitLossNumbers = tradelist[(tradelist.exit_label == "time_exit_rules_loss_long") | (tradelist.exit_label == "time_exit_rules_loss_short")].exit_label.count()  
        if TimeExitLossNumbers != 0:
            print("Time Exit Loss Number:   ", TimeExitLossNumbers, "equivalent to",  
                  round(TimeExitLossNumbers / operation_number(tradelist.operations) * 100, 2), "%")

        TimeExitGainNumbers = tradelist[(tradelist.exit_label == "time_exit_rules_gain_long") | (tradelist.exit_label == "time_exit_rules_gain_short")].exit_label.count()  
        if TimeExitGainNumbers != 0:
            print("Time Exit Gain Number:   ", TimeExitGainNumbers, "equivalent to",  
                  round(TimeExitGainNumbers / operation_number(tradelist.operations) * 100, 2), "%")
            
        MoneyStopLosses = tradelist[(tradelist.exit_label == "money_stoploss_long") |\
                                    (tradelist.exit_label == "money_stoploss_short") |\
                                    (tradelist.exit_label == "money_stoploss_entrybar_long") |\
                                    (tradelist.exit_label == "money_stoploss_entrybar_short")].exit_label.count()  
        
        MoneyStopLossesEntryBar = tradelist[(tradelist.exit_label == "money_stoploss_entrybar_long") |\
                                            (tradelist.exit_label == "money_stoploss_entrybar_short")].exit_label.count()  
        
        if MoneyStopLosses != 0:
            print("Money Stoploss Number:   ", MoneyStopLosses, "equivalent to",  
                  round(MoneyStopLosses / operation_number(tradelist.operations) * 100, 2), "%",
                 Fore.LIGHTWHITE_EX + "(on entry bar:", round(MoneyStopLossesEntryBar / operation_number(tradelist.operations) * 100, 2), "%)" + Fore.RESET)

        MoneyTargets = tradelist[(tradelist.exit_label == "money_target_long") |\
                                 (tradelist.exit_label == "money_target_short") |\
                                 (tradelist.exit_label == "money_target_entrybar_long") |\
                                 (tradelist.exit_label == "money_target_entrybar_short")].exit_label.count()  
        
        MoneyTargetsEntryBar = tradelist[(tradelist.exit_label == "money_target_entrybar_long") |\
                                         (tradelist.exit_label == "money_target_entrybar_short")].exit_label.count() 
        if MoneyTargets != 0:
            print("Money Target Number:     ", MoneyTargets, "equivalent to", 
                  round(MoneyTargets / operation_number(tradelist.operations) * 100, 2), "%",
                 Fore.LIGHTWHITE_EX + "(on entry bar:", round(MoneyTargetsEntryBar / operation_number(tradelist.operations) * 100, 2), "%)" + Fore.RESET)
          
        PercentStopLosses = tradelist[(tradelist.exit_label == "percent_stoploss_long") |\
                                      (tradelist.exit_label == "percent_stoploss_short") |\
                                      (tradelist.exit_label == "percent_stoploss_entrybar_long") |\
                                      (tradelist.exit_label == "percent_stoploss_entrybar_short")].exit_label.count()  

        PercentStopLossesEntryBar = tradelist[(tradelist.exit_label == "percent_stoploss_entrybar_long") |\
                                              (tradelist.exit_label == "percent_stoploss_entrybar_short")].exit_label.count()  
        
        if PercentStopLosses != 0:
            print("Percent Stoploss Number: ", PercentStopLosses, "equivalent to",  
                  round(PercentStopLosses / operation_number(tradelist.operations) * 100, 2), "%",
                  Fore.LIGHTWHITE_EX + "(on entry bar:", round(PercentStopLossesEntryBar / operation_number(tradelist.operations) * 100, 2), "%)" + Fore.RESET)

        PercentTarget = tradelist[(tradelist.exit_label == "percent_target_long") |\
                                  (tradelist.exit_label == "percent_target_short") |\
                                  (tradelist.exit_label == "percent_target_entrybar_long") |\
                                  (tradelist.exit_label == "percent_target_entrybar_short")].exit_label.count()  

        PercentTargetEntryBar = tradelist[(tradelist.exit_label == "percent_target_entrybar_long") |\
                                          (tradelist.exit_label == "percent_target_entrybar_short")].exit_label.count()
        
        if PercentTarget != 0:
            print("Percent Target Number:   ", PercentTarget, "equivalent to",  
                  round(PercentTarget / operation_number(tradelist.operations) * 100, 2), "%",
                  Fore.LIGHTWHITE_EX + "(on entry bar:", round(PercentTargetEntryBar / operation_number(tradelist.operations) * 100, 2), "%)" + Fore.RESET)
     
        StopLevel = tradelist[(tradelist.exit_label == "stop_level_long") |\
                              (tradelist.exit_label == "stop_level_short") |\
                              (tradelist.exit_label == "stop_level_entrybar_long") |\
                              (tradelist.exit_label == "stop_level_entrybar_short")].exit_label.count()  

        StopLevelEntryBar = tradelist[(tradelist.exit_label == "stop_level_entrybar_long") |\
                              (tradelist.exit_label == "stop_level_entrybar_short")].exit_label.count()
        
        if StopLevel != 0:
            print("Stop Level Number:       ", StopLevel, "equivalent to", 
                  round(StopLevel / operation_number(tradelist.operations) * 100, 2), "%",
                  Fore.LIGHTWHITE_EX + "(on entry bar:", round(StopLevelEntryBar / operation_number(tradelist.operations) * 100, 2), "%)" + Fore.RESET)
        
        TargetLevel = tradelist[(tradelist.exit_label == "target_level_long") |\
                                (tradelist.exit_label == "target_level_short") |\
                                (tradelist.exit_label == "target_level_entrybar_long") |\
                                (tradelist.exit_label == "target_level_entrybar_short")].exit_label.count() 
        
        TargetLevelEntryBar = tradelist[(tradelist.exit_label == "target_level_entrybar_long") |\
                                        (tradelist.exit_label == "target_level_entrybar_short")].exit_label.count()

        if TargetLevel != 0:
            print("Target Level Number:     ", TargetLevel, "equivalent to", 
                  round(TargetLevel / operation_number(tradelist.operations) * 100, 2), "%",
                  Fore.LIGHTWHITE_EX + "(on entry bar:", round(TargetLevelEntryBar / operation_number(tradelist.operations) * 100, 2), "%)" + Fore.RESET)
            
        print("")
        print(Fore.LIGHTGREEN_EX + "*****************************************************************************************")
        print("********* Copyright 2022 - by Gandalf Project R&D - Coded by Giovanni Trombetta *********")
        print("*****************************************************************************************" + Fore.RESET)

        if interactive == False:
            plot_equity(open_equity,"green")
            plot_equity_price(dataset, open_equity)
            plot_drawdown(open_equity,"red")
            plot_drawdown_perc(open_equity,initial_capital,"red")
            plot_capital(tradelist,margin_percent)
            plot_annual_histogram(tradelist.operations)
            plot_monthly_bias_histogram(tradelist.operations)
            plot_equity_heatmap(tradelist.operations,False)
            plot_mae(tradelist)
            plot_mfe(tradelist)
            plot_trade_duration(tradelist)
        else:
            plot_equity_interactive(open_equity,"green")
            plot_drawdown_interactive(open_equity)
            plot_drawdown_perc_interactive(open_equity,initial_capital)
            plot_capital_interactive(tradelist)
            plot_annual_histogram_interactive(tradelist.operations)
            plot_monthly_bias_histogram_interactive(tradelist.operations)
            plot_equity_heatmap_interactive(tradelist.operations,False)
            plot_mae(tradelist)
            plot_mfe(tradelist)
            plot_trade_duration(tradelist)

        return

# GENERIC FUNCTIONS *******************************************************************************************************start

def crossover(array1, array2):
    return (array1 > array2) & (array1.shift(1) < array2.shift(1))

def crossunder(array1, array2):
    return (array1 < array2) & (array1.shift(1) > array2.shift(1))

def highest(*args):
    maximum = 0
    for arg in args:
        if arg > maximum:
            maximum = arg
    return maximum

def highest_serie(*args):
    maximum = args[0]
    for arg in args:
        maximum = np.where(arg > maximum, arg, maximum)
    return maximum

def lowest(*args):
    minimum = np.inf
    for arg in args:
        if arg < minimum:
            minimum = arg
    return minimum

def lowest_serie(*args):
    minimum = args[0]
    for arg in args:
        minimum = np.where(arg < minimum, arg, minimum)
    return minimum

def tick_correction_up(level,tick):
    if level != level:
        level = 0
    multiplier = math.ceil(level/tick)
    return multiplier * tick

def tick_correction_down(level,tick):
    if level != level:
        level = 0
    multiplier = math.floor(level/tick)
    return multiplier * tick

def stop_check(dataframe,rules,level,direction):
    """
    Funzione per validare una regola di ingresso o di uscita rispetto ad un setup stop
    Viene verificata il superamento del massimo (long) o minimo (short) sul level
    """
    service_dataframe = pd.DataFrame(index = dataframe.index)
    service_dataframe['rules'] = rules
    service_dataframe['level'] = level
    service_dataframe['low'] = dataframe.low
    service_dataframe['high'] = dataframe.high

    if direction == "long":
        service_dataframe['new_rules'] = np.where((service_dataframe.rules == True) &\
                                                  (service_dataframe.high.shift(-1) >= service_dataframe.level.shift(-1)), 
                                                  True, False)
    if direction == "short":
        service_dataframe['new_rules'] = np.where((service_dataframe.rules == True) &\
                                                  (service_dataframe.low.shift(-1) <= service_dataframe.level.shift(-1)), 
                                                  True, False)
    return service_dataframe.new_rules

def limit_check(dataframe,rules,level,direction):
    """
    Funzione per validare una regola di ingresso o di uscita rispetto ad un setup limit
    Viene verificata il raggiungimento del minimo (long) o massimo (short) sul level
    """
    service_dataframe = pd.DataFrame()
    service_dataframe['rules'] = rules
    service_dataframe['level'] = level
    service_dataframe['low'] = dataframe.low
    service_dataframe['high'] = dataframe.high
    
    if direction == "long":
        service_dataframe['new_rules'] = np.where((service_dataframe.rules == True) & \
                                                  (service_dataframe.low.shift(-1) <= service_dataframe.level.shift(-1)), 
                                                  True, False)
    if direction == "short":
        service_dataframe['new_rules'] = np.where((service_dataframe.rules == True) &
                                                  (service_dataframe.high.shift(-1) >= service_dataframe.level.shift(-1)), 
                                                  True, False)
    return service_dataframe.new_rules

def marketposition_generator(enter_rules,exit_rules):
    """
    Funzione per calcolare il marketposition date due serie di enter_rules and exit_rules
    """
    service_dataframe = pd.DataFrame(index = enter_rules.index)
    service_dataframe['enter_rules'] = enter_rules
    service_dataframe['exit_rules'] = exit_rules
    
    status = 0
    mp = []
    for (i, j) in zip(enter_rules, exit_rules):
        if status == 0:
            if i == 1 and j != -1:
                status = 1
        else:
            if j == -1:
                status = 0
        mp.append(status)
        
    service_dataframe['mp_new'] = mp
    service_dataframe.mp_new = service_dataframe.mp_new.shift(1)
    service_dataframe.iloc[0,2] = 0
    #service_dataframe.to_csv("marketposition_generator.csv")
    return service_dataframe.mp_new

def barssinceentry(enter_rules,exit_rules):
    """
    Funzione per calcolare da quante barre sia in corso l'operazione
    date due serie di enter_rules and exit_rules
    """
    service_dataframe = pd.DataFrame(index = enter_rules.index)
    service_dataframe['enter_rules'] = enter_rules.apply(lambda x: 1 if x == True else 0)
    service_dataframe['exit_rules'] = exit_rules.apply(lambda x: -1 if x == True else 0)
    
    service_dataframe["mp"] = marketposition_generator(service_dataframe.enter_rules,
                                                       service_dataframe.exit_rules)

    service_dataframe["service1"] = service_dataframe.mp.cumsum() * service_dataframe.mp
    service_dataframe["service2"] = np.where(service_dataframe.mp.shift(1) == 0, service_dataframe.service1, np.nan)
    service_dataframe["service2"] = service_dataframe["service2"].ffill()
    service_dataframe["barssinceentry"] = service_dataframe["mp"] * abs(service_dataframe["service1"] - service_dataframe["service2"] + 1)

    return service_dataframe["barssinceentry"].fillna(0)

def spill_capital_from_log(filename,name):
    """
    Function to extract capital from log
    """
    log = pd.read_csv(filename, sep = " ")
    log["conversion_dates"] = log.dates.apply(lambda x: pd.Timestamp(x))
    log.set_index(log.conversion_dates, inplace = True)
    log[name] = log.entry_price * log.shares
    log = log.iloc[:,-1]
    return log

def create_entries_exits(dataset,tradelist):
    """
    Function to extract entries and exits signals from tradelist
    to plot on the asset graph
    """
    entry_signals = tradelist.copy().set_index("entry_date")
    entry_signals.drop(["exit_date","exit_label","exit_price","bars_in_trade",
                        "mae","mfe","operations"], axis = 1, inplace = True)
    entry_signals = entry_signals.resample('H').sum().replace(0,np.nan)
    
    exit_signals = tradelist.copy()
    exit_signals.drop(["id","entry_date","entry_label","quantity",
                    "entry_price","capital"], axis = 1, inplace = True)
    exit_signals = exit_signals.resample('H').sum().replace(0,np.nan)
    
    asset = dataset.copy().iloc[:,:5]
    
    signals = pd.concat([asset,entry_signals,exit_signals], axis = 1)
    
    signals["entries"] = np.where(signals.capital == signals.capital,1,np.nan)
    signals["exits"] = np.where(signals.operations == signals.operations,1,np.nan)
    
    return signals

def match_tradelist_dataset(my_dataset,my_tradelist):
    """
    Function to extract entries and exits signals from tradelist
    to copy on dataset dataframe
    """
    tradelist_dataset = my_dataset.copy()
    tradelist_dataset["date"] = tradelist_dataset.index

    entries = list(my_tradelist.entry_date)
    exits = list(my_tradelist.exit_date)
    entryprices = list(my_tradelist.entry_price)
    exitprices = list(my_tradelist.exit_price)

    entry_event = []
    exit_event = []

    for i in range(len(tradelist_dataset)):
        if tradelist_dataset.date[i] in entries:
            entry_event.append(1)
        else:
            entry_event.append(0)
        if tradelist_dataset.date[i] in exits:
            exit_event.append(1)
        else:
            exit_event.append(0)
            
    tradelist_dataset["entries"] = entry_event
    tradelist_dataset["exits"] = exit_event
    tradelist_dataset["op_number"] = tradelist_dataset.entries.cumsum()

    #tradelist_dataset["entries"] = np.where([tradelist_dataset.date[i] in entries for i in range(len(tradelist_dataset))],1,0)
    #tradelist_dataset["exits"] = np.where([tradelist_dataset.date[i] in exits for i in range(len(tradelist_dataset))],1,0)
        
    return tradelist_dataset

def plot_operations_basic(tradelist_dataset,startdate,enddate):
    """
    Function to plot entry and exit line
    """
    tradelist_to_plot = tradelist_dataset.loc[startdate:enddate]
    fig = plt.figure(figsize = (18,10))
    plt.plot(tradelist_to_plot.close, color="grey", lw=2.)

    plt.plot(tradelist_to_plot[tradelist_to_plot.entries == 1].index, 
             tradelist_to_plot["low"][tradelist_to_plot.entries == 1], 
             "^", markersize = 10, color = "green", label = 'buy')
    plt.plot(tradelist_to_plot[tradelist_to_plot.exits == 1].index, 
             tradelist_to_plot["high"][tradelist_to_plot.exits == 1], 
             "v", markersize = 10, color = "red", label = 'sell')
    plt.legend()
    plt.grid()
    plt.show()
    return

def plot_operations(mixed_dataset,startdate,enddate,volumes,ts_tradelist,tradedir):
    """
    Function to plot entry and exit candles
    """

    import mplfinance as mpl

    tradelist_to_plot = mixed_dataset.loc[startdate:enddate]

    buy_signals = []
    sell_signals = []
    sellshort_signals = []
    buytocover_signals = []
    
    entryprice = []
    exitprice = []

    for i in range(len(tradelist_to_plot)):
        if (tradelist_to_plot.entries.iloc[i] == 1):
            if tradedir.lower() == "long":
                buy_signals.append(tradelist_to_plot.iloc[i]["low"] * 0.999)
            elif tradedir.lower() == "short":
                sellshort_signals.append(tradelist_to_plot.iloc[i]["high"] * 1.001)
            #print(tradelist_to_plot.op_number[i])
            entryprice.append(ts_tradelist.iloc[tradelist_to_plot.op_number[i-1]].entry_price)
            ep = ts_tradelist.iloc[tradelist_to_plot.op_number[i-1]].exit_price
            #print(entryprice[-1])
            #print(exitprice[-1])
        else:
            buy_signals.append(np.nan)
            entryprice.append(np.nan)
            sellshort_signals.append(np.nan)
            
        if (tradelist_to_plot.exits.iloc[i] == 1):
            if tradedir.lower() == "long":
                sell_signals.append(tradelist_to_plot.iloc[i]["high"] * 1.001)
            elif tradedir.lower() == "short":
                buytocover_signals.append(tradelist_to_plot.iloc[i]["low"] * 0.999)
            if 'ep' in locals():
                exitprice.append(ep)
            else:
                exitprice.append(np.nan)
        else:
            sell_signals.append(np.nan)
            exitprice.append(np.nan)
            buytocover_signals.append(np.nan)

    if tradedir.lower() == "long":
        
        buy_markers = mpl.make_addplot(buy_signals, type='scatter', markersize=60, marker='^', color = "green")
        sell_markers = mpl.make_addplot(sell_signals, type='scatter', markersize=60, marker='v', color = "red")

        entryprice_markers = mpl.make_addplot(entryprice, type='scatter', markersize=60, marker='.', color = "green")
        exitprice_markers = mpl.make_addplot(exitprice, type='scatter', markersize=60, marker='.', color = "red")
        
        signals = [buy_markers, sell_markers, entryprice_markers, exitprice_markers]
    
    elif tradedir.lower() == "short":
        
        sellshort_markers = mpl.make_addplot(sellshort_signals, type='scatter', markersize=60, marker='v', color = "red")
        buytocover_markers = mpl.make_addplot(buytocover_signals, type='scatter', markersize=60, marker='^', color = "green")

        entryprice_markers = mpl.make_addplot(entryprice, type='scatter', markersize=60, marker='.', color = "red")
        exitprice_markers = mpl.make_addplot(exitprice, type='scatter', markersize=60, marker='.', color = "green")

        signals = [sellshort_markers, buytocover_markers, entryprice_markers, exitprice_markers]
        
    mpl.plot(tradelist_to_plot, 
             type = "candle",
             style = "classic",
             #title='prices',
             #ylabel='Price ($)',
             volume = volumes,
             show_nontrading = False,
             figratio = (18,10),
             figscale = 1.5,
             addplot = signals)
    return

# GENERIC FUNCTIONS *********************************************************************************************************end

# VALIDATION FUNCTIONS ****************************************************************************************************start

def gsa_advanced(dataframe,step_IS,step_OOS):
    """
    Funzione che genera due dataframe ottenuti a partire dal dataframe totale
    altenando periodi di in sample ed out of sample in maniera simmetrica o asimmetrica
    ES: step_IS = 100, step_OOS = 100 barre -> IS 100 barre OOS 100 barre alternate
    ES: step_IS = 100, step_OOS = 50 barre -> IS 100 barre OOS 50 barre alternate
    """
    
    start_period_days_IS = []
    end_period_days_IS = []
    start_period_days_OOS = []
    end_period_days_OOS = []
    
    counter = 1
    start = 0
    stop = start + step_IS
    while start < len(dataframe):
        #print(start, stop, start + step_IS, stop + step_OOS)
        if start == 0:
            #print("period:", counter)
            dataset_IS = dataframe.copy().iloc[start:stop]
            #print("dataset_IS", len(dataset_IS))
            start_period_days_IS.append(dataset_IS.index[0])
            end_period_days_IS.append(dataset_IS.index[-1])
            #print(start_period_days_IS, end_period_days_IS)
            
            dataset_OOS = dataframe.copy().iloc[start + step_IS:stop + step_OOS]
            #print("dataset_OOS", len(dataset_OOS))
            start_period_days_OOS.append(dataset_OOS.index[0])
            end_period_days_OOS.append(dataset_OOS.index[-1])
            #print(start_period_days_OOS, end_period_days_OOS)
        else:
            #print("period:", counter)
            block_IS = dataframe.copy().iloc[start:stop]
            block_OOS = dataframe.copy().iloc[start + step_IS:stop + step_OOS]
            #print("block_IS", len(block_IS))
            #print("block_OOS", len(block_OOS))
            dataset_IS = pd.concat([dataset_IS, block_IS])
            start_period_days_IS.append(block_IS.index[0])
            end_period_days_IS.append(block_IS.index[-1])
            #print(start_period_days_IS, end_period_days_IS)
            
            dataset_OOS = pd.concat([dataset_OOS, block_OOS])
            if len(block_OOS) != 0:
                start_period_days_OOS.append(block_OOS.index[0])
                end_period_days_OOS.append(block_OOS.index[-1])
                #print(start_period_days_OOS, end_period_days_OOS)
        counter += 1
            
        #print(len(dataset_IS),len(dataset_OOS))
        start += step_IS + step_OOS
        stop = start + step_IS
    return dataset_IS, dataset_OOS,\
           start_period_days_IS, end_period_days_IS,\
           start_period_days_OOS, end_period_days_OOS

def gsa_period_setup(dataframe, start_period, end_period):
    service = dataframe.copy()
    for i in range(len(start_period)):
        field_name = "setup_period_" + str(i)
        #print(field_name)
        service[field_name] = np.where((service.index >= start_period[i]) &\
                                       (service.index <= end_period[i]),
                                       1,0)
    service["setup_period"] = service.iloc[:,-len(start_period):].sum(axis = 1)
    return service["setup_period"]

def query(dataframe, optimization_fitness, x, y):
    return dataframe[(dataframe["opt1"] == x) & (dataframe["opt2"] == y)][optimization_fitness]

def double_parameter_optimization_graph(optimization_report, optimization_fitness):
    
    import plotly.graph_objs as go

    x = []
    y = []
    z = []
    
    x = optimization_report["opt1"]
    y = optimization_report["opt2"]
    z = query(optimization_report, optimization_fitness, x, y)
    
    df = pd.DataFrame()
    df["opt1"] = x
    df["opt2"] = y
    df["optimization_fitness"] = z
    
    a = df.pivot('opt1', 'opt2', 'optimization_fitness')

    surface = go.Surface(y = a.index,
                         x = a.columns,
                         z = a.values,
                         colorscale = 'RdBu')
    
    data = [surface]
    
    layout = go.Layout(title = '3D Optimization Report - by Gandalf Project R&D',
                       scene = dict(xaxis = dict(title = 'Opt2'),
                                    yaxis = dict(title = 'Opt1'),
                                    zaxis = dict(title = optimization_fitness)),
                       autosize = False,
                       width = 800,
                       height = 600,
                       margin = dict(l = 0, r = 0, b = 50, t = 50))  
        
    fig = go.Figure(data = data, layout = layout)
    
    #fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))

    iplot(fig)
    
    return 

def plot_trades_distribution(trading_system, bins_divisor, color):
    """
    Funzione per graficare la distribuzione di trade
    """
    ops = trading_system.operations.dropna()
    print("Percentiles of Trades Distribution")
    print(ops.describe(percentiles = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]))

    plt.figure(figsize = (10,5), dpi = 300)
    sns.distplot(ops, bins = int(ops.count() / bins_divisor), 
                 color = color, label = "Operations")
    plt.title("Distribution of Operations")
    plt.xlabel("Profit/Loss")
    plt.ylabel("Percentage of Operations")
    plt.legend()
    plt.grid()
    plt.show()
    return

def plot_validation_distribution(operations_IS, operations_OOS, bins_divider):
    """
    Funzione per sovrapporre le distribuzioni dei trade in In Sample con quelli in Out of Sample
    """
    plt.figure(figsize = (10,5), dpi = 300)
    sns.distplot(operations_IS.dropna(), bins = int(operations_IS.count() / bins_divider), 
                 color = "red", label = "In Sample Operations")
    sns.distplot(operations_OOS.dropna(), bins = int(operations_OOS.count() / bins_divider), 
                 color = "green", label = "Out of Sample Operations")
    plt.title("In Sample vs Out of Sample Operations Distribution")
    plt.xlabel("Profit/Loss")
    plt.ylabel("Percentage of Operations")
    plt.legend()
    plt.grid()
    plt.show()
    return

def validation_performances(operations_IS, open_equity_IS, operations_OOS, open_equity_OOS):
    """
    Funzione che calcola le metriche di confronto tra In Sample ed Out Of Sample
    """
    profit_IS = profit(open_equity_IS)
    profit_OOS = profit(open_equity_OOS)
    nop_IS = operation_number(operations_IS)
    nop_OOS = operation_number(operations_OOS)
    at_IS = avg_trade(operations_IS)
    at_OOS = avg_trade(operations_OOS)
    pf_IS = profit_factor(operations_IS)
    pf_OOS = profit_factor(operations_OOS)
    pw_IS = percent_win(operations_IS)
    pw_OOS = percent_win(operations_OOS)
    rrr_IS = reward_risk_ratio(operations_IS)
    rrr_OOS = reward_risk_ratio(operations_OOS)
    add_IS = avgdrawdown_nozero(open_equity_IS)
    add_OOS = avgdrawdown_nozero(open_equity_OOS)
    mdd_IS = max_draw_down(open_equity_IS)
    mdd_OOS = max_draw_down(open_equity_OOS)

    print("In Sample vs Out of Sample Statistics - by Gandalf Project R&D")
    print("")
    print("Profit:                 [IS]", profit_IS, "[OOS]", profit_OOS, 
          "-> delta:", round((profit_OOS - profit_IS)/profit_OOS * 100), "%")
    print("Operations:             [IS]", nop_IS, "[OOS]", nop_OOS, 
          "-> delta:", round((nop_OOS - nop_IS)/nop_OOS * 100), "%")
    print("")
    print("Average Trade:          [IS]", at_IS, "[OOS]", at_OOS, 
          "-> delta:", round((at_OOS - at_IS)/at_OOS * 100), "%")
    print("Profit Factor:          [IS]", pf_IS, "[OOS]", pf_OOS, 
          "-> delta:", round((pf_OOS - pf_IS)/pf_OOS * 100), "%")
    print("Percent Winning Trades: [IS]", pw_IS, "[OOS]", pw_OOS, 
          "-> delta:", round((pw_OOS - pw_IS)/pw_OOS * 100), "%")
    print("Reward Risk Ratio:      [IS]", rrr_IS, "[OOS]", rrr_OOS, 
          "-> delta:", round((rrr_OOS - rrr_IS)/rrr_OOS * 100), "%")
    print("Avg Open Draw Down:     [IS]", add_IS, "[OOS]", add_OOS, 
          "-> delta:", round((add_OOS - add_IS)/add_OOS * 100), "%")
    print("Max Open Draw Down:     [IS]", mdd_IS, "[OOS]", mdd_OOS, 
          "-> delta:", round((mdd_OOS - mdd_IS)/mdd_OOS * 100), "%")
    return

def plot_rectangles(dataset, start, step1, step2, color):
    while start < len(dataset):
        stop = start + step1
        if start <= len(dataset):
            start_parsed = dataset.index[start]
        else:
            start_parsed = dataset.index[len(dataset) - 1]
        if stop <= len(dataset):
            end_parsed = dataset.index[stop]
        else:
            end_parsed = dataset.index[len(dataset) - 1]
        print(start, start_parsed, stop, end_parsed)
        plt.axvline(x = start_parsed, color = color)
        plt.axvline(x = end_parsed, color = color)
        plt.axvspan(start_parsed, end_parsed, facecolor = color, alpha = 0.4)
        start += step1 + step2

def plot_periods(dataset, equity, step_IS, step_OOS):
    
    from pandas.plotting import register_matplotlib_converters
    
    new_highs = equity.expanding().max()
    limes = pd.DataFrame(np.where(equity == new_highs, new_highs, np.nan), index = equity.index)
    
    plt.figure(figsize = (16, 8), dpi = 300)

    # Generazione rettangoli In Sample
    print("In Sample:")
    plot_rectangles(dataset, 0, step_IS, step_OOS, "yellow")
    print("")

    # Generazione rettangoli Out of Sample
    print("Out of Sample:")
    plot_rectangles(dataset, step_IS, step_OOS, step_IS, "springgreen")
    plt.plot(equity, color='green')
    plt.plot(limes, color = "lime", marker =".", markersize = 6)
    plt.xlabel("Time")
    plt.ylabel("Profit/Loss")
    plt.title("Equity Lines")
    # - Powered by Gandalf Project"
    plt.xticks(rotation="vertical")
    plt.grid(True)
    plt.show()
    
def plot_periods_double(dataset, equity, step_IS, step_OOS):
    
    from pandas.plotting import register_matplotlib_converters
    
    new_highs = equity.expanding().max()
    limes = pd.DataFrame(np.where(equity == new_highs, new_highs, np.nan), index = equity.index)
    
    fig, ax1 = plt.subplots(figsize = [12, 8], dpi = 300)
    #ax1.figure.set_size_inches(12, 8)

    # Generazione rettangoli In Sample
    print("In Sample:")
    plot_rectangles(dataset, 0, step_IS, step_OOS, "yellow")
    print("")

    # Generazione rettangoli Out of Sample
    print("Out of Sample:")
    plot_rectangles(dataset, step_IS, step_OOS, step_IS, "springgreen")
    plt.xticks(rotation = "vertical")
    plt.grid(True)
    
    ax1.set_title("Strategy Validation Architecture ")
    #- Powered by Gandalf Project
    ax1.plot(dataset.close, color = 'tan', lw = 2, alpha = 0.5)
    #ax1.legend(["Asset"])
    #ax1.fill_between(dataset.index, 0, dataset.close, alpha = 0.3)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Asset Price")
    
    ax2 = ax1.twinx()
    ax2.plot(equity, color='green')
    ax2.plot(limes, color = "lime", marker =".", markersize = 6)
    ax2.set_ylabel("Strategy Profit/Loss")
    ax2.legend(["Equity Line"])

    plt.show()
    
def gpdr(operationsIS, operationsOoS, step, tolerance):
    """
     Gandalf Persistence Distribution Ratio (GPDR)
     Versione evoluta: dividiamo i percentili in step blocchi e per ciascuno confrontiamo il valore numerico
     in In Sample ed in Out of Sample. Tolleriamo un degrado tra risultati In Sample ed Out of Sample
     di non oltre la tolerance.
    """
    
    #step = 0.1
    values = np.arange(0.0, 1.0 + step, step)
    print(values)
    print("")

    IS = 0
    OoS = 0
    for i in values:
        if operationsIS.quantile(i) > 0:
            if ((1 - tolerance) * operationsIS.quantile(i)) <= operationsOoS.quantile(i):
                OoS += 1
            else:
                IS += 1
            print(round(i,2), 
                  round(operationsIS.quantile(i), 2), 
                  round((1 - tolerance) * operationsIS.quantile(i), 2),
                  round(operationsOoS.quantile(i), 2), IS, OoS, )
        if operationsIS.quantile(i) <= 0:
            if ((1 + tolerance) * operationsIS.quantile(i)) <= operationsOoS.quantile(i):
                OoS += 1
            else:
                IS += 1
            print(round(i,2),
                  round(operationsIS.quantile(i), 2),
                  round((1 + tolerance) * operationsIS.quantile(i), 2),
                  round(operationsOoS.quantile(i), 2), IS, OoS, )
    
    best = round(OoS / len(values) * 100, 2)
    worst = round(IS / len(values) * 100, 2)
    
    print("")
    if worst != 0:
        print("Gandalf Persistence Distribution Ratio (GPDR):", round(best / worst,2))
    else:
        print("Gandalf Persistence Distribution Ratio (GPDR):", np.inf)
    print("Gandalf Persistence Distribution Index (GPDI_OOS):", best, "%",)
    print("Gandalf Persistence Distribution Index (GPDI_IS):", worst, "%")
    return

# VALIDATION FUNCTIONS ******************************************************************************************************end

# OPTIMIZATION FUNCTIONS **************************************************************************************************start

def substitution(element,step_values):
    """
    Function to map percentiles classes on each matrix element
    """
    #print(element)
    conditions = []
    #print(step_values)
    for i in range(1,len(step_values)):
        if i > 0:
            if i == 1:
                conditions.append(element >= step_values[i-1] and element <= step_values[i])
            else:
                #print(element, step_values[i-1], step_values[i], element > step_values[i-1] and element <= step_values[i])
                conditions.append(element > step_values[i-1] and element <= step_values[i])
    #print(conditions)
    element_class = int(np.where(conditions)[0]) + 1
    #print(element_class)
    return element_class

def classification(matrix, slices):
    """
    Function to transform original optimization matrix
    in a percentiles classification matrix
    """
    new_matrix = matrix.copy()
    new_matrix_flatten = new_matrix.values
    distribution_slices = np.arange(slices + 1)
    #print(distribution_slices)
    
    maximum = np.max(new_matrix_flatten)
    minimum = np.min(new_matrix_flatten)
    interval = maximum - minimum
    if slices != 0:
        step = interval / slices
    else:
        print("")
        return
    #print(maximum, minimum, step)

    step_values = []
    for el in distribution_slices:
        step_values.append(minimum + el * step)
        if el == distribution_slices[-1]:
            step_values.append(minimum + el * step + 0.0001)
        #print(el, step_values[-1])
        #print(step_values)
    print("")
    
    new_matrix = matrix.applymap(lambda x: substitution(x,step_values))
    return new_matrix

def substitution_percentile(element,percentiles):
    """
    Function to map percentiles classes on each matrix element
    """
    #print(element)
    conditions = []
    #print(percentiles)
    for i in range(len(percentiles)):
        if i > 0:
            if i == 1:
                conditions.append(element >= percentiles[i-1] and element <= percentiles[i])
            else:
                conditions.append(element > percentiles[i-1] and element <= percentiles[i])
    #print(conditions)
    element_class = int(np.where(conditions)[0]) + 1
    #print(element_class)
    return element_class

def classification_percentile(matrix, slices):
    """
    Function to transform original optimization matrix
    in a percentiles classification matrix
    """
    new_matrix = matrix.copy()
    new_matrix_flatten = new_matrix.values
    distribution_slices = np.linspace(0, 100, slices + 1)
    percentiles = []
    for el in distribution_slices:
        percentiles.append(np.percentile(new_matrix_flatten, el))
        #print(el, percentiles[-1])
    print("")
    
    new_matrix = matrix.applymap(lambda x: substitution_percentile(x,percentiles))
    return new_matrix

def double_parameter_classes_graph(classes_matrix, optimization_fitness):
    
    import plotly.graph_objs as go

    surface = go.Surface(y = classes_matrix.index,
                         x = classes_matrix.columns,
                         z = classes_matrix.values,
                         colorscale = 'RdBu')
    
    data = [surface]
    
    layout = go.Layout(title = 'Classes 3D Optimization Report - by Gandalf Project R&D',
                       scene = dict(xaxis = dict(title = 'Opt2'),
                                    yaxis = dict(title = 'Opt1'),
                                    zaxis = dict(title = optimization_fitness)),
                       autosize = False,
                       width = 800,
                       height = 600,
                       margin = dict(l = 0, r = 0, b = 50, t = 50))  
        
    fig = go.Figure(data = data, layout = layout)
    
    #fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))

    iplot(fig)
    
    return

def evaluate_point(matrix,row,column,tolerance):
    vote = 0
    if row == 0 or row == len(matrix) - 1 or column == 0 or column == len(matrix[0]) - 1:
        return 0
    else:
        reference_class = matrix[row, column]
        
        # sud
        if (matrix[row + 1, column] == reference_class) or\
           (reference_class - tolerance <= matrix[row + 1, column] <= reference_class + tolerance):
            vote += 1
            #print("sud:", vote)
        # nord
        if (matrix[row - 1, column] == reference_class) or\
           (reference_class - tolerance <= matrix[row - 1, column] <= reference_class + tolerance):
            vote += 1
            #print("nord:", vote)
        # est
        if (matrix[row, column + 1] == reference_class) or\
           (reference_class - tolerance <= matrix[row, column + 1] <= reference_class + tolerance):
            vote += 1
            #print("est:", vote)
        # ovest
        if (matrix[row, column - 1] == reference_class) or\
           (reference_class - tolerance <= matrix[row, column - 1] <= reference_class + tolerance):
            vote += 1
            #print("ovest:", vote)
            
        # diagonal: sud-est
        if (matrix[row + 1, column + 1] == reference_class) or\
           (reference_class - tolerance <= matrix[row + 1, column + 1] <= reference_class + tolerance):
            vote += 1
            #print("sud-est:", vote)
        # diagonal: nord-est
        if (matrix[row - 1, column + 1] == reference_class) or\
           (reference_class - tolerance <= matrix[row - 1, column + 1] <= reference_class + tolerance):
            vote += 1
            #print("nord-est:", vote)
        # diagonal: nord-ovest
        if (matrix[row - 1, column - 1] == reference_class) or\
           (reference_class - tolerance <= matrix[row - 1, column - 1] <= reference_class + tolerance):
            vote += 1
            #print("nord-ovest:", vote)
        # diagonal: sud-ovest
        if (matrix[row + 1, column - 1] == reference_class) or\
           (reference_class - tolerance <= matrix[row + 1, column - 1] <= reference_class + tolerance):
            vote += 1
            #print("sud-ovest:", vote)
            
        return vote
    
def X_graph(stability_matrix):
    return stability_matrix[stability_matrix == 8].fillna(".").replace(8,"X")

def boolean_matrix(stability_matrix):
    return stability_matrix[stability_matrix == 8].fillna(0).replace(8,1)

def X_graph_from_boolean(stability_matrix):
    return stability_matrix[stability_matrix == 1].replace(np.nan,".").replace(1,"X")

def single_opt_stable_points(matrix, metric, slices, tolerance,
                             algorithm, surface_plot,verbose):
    """
    Function to obtain stable points on single fitness 3D surface
    
    INPUTS
    matrix: optimization results from a double parameter opt
    metric: fitness to calculate z axis
    slices: number of samples from lowest to highest value into the optimization results
    tolerance: percentage difference between class of each point and class of neighboring points
    algorithm: type of algorithm to generate classes 
               "normalizer": division into "slices" classes the interval between maximum and minimum value 
                             of the optimization results
               "percentiles": division into "slices" classes percentiles values
                             of the optimization results
    surface_plot: plotting of 3D surface
                  True: will plot 3D surface
                  False: won't plot 3D surface
                             
    OUTPUTS
    opt_matrix: pivot of matrix
    opt_matrix_classified: classified opt_matrix
    stability_matrix: enhancement of opt_matrix_classified with stability values (0 to 8)
    inner_results: pandas dataframe with best solutions ranked
    """
    optimization_fitness = metric
    inner_tolerance = int(slices * tolerance / 100)

    x = []
    y = []
    z = []

    x = matrix["opt1"]
    y = matrix["opt2"]
    z = query(matrix, optimization_fitness, x, y)

    df = pd.DataFrame()
    df["opt1"] = x
    df["opt2"] = y
    df["optimization_fitness"] = z

    opt_matrix = df.pivot('opt1', 'opt2', 'optimization_fitness')
    if verbose == True:
        print(opt_matrix)
    
    if algorithm == "normalizer":
        opt_matrix_classified = classification(opt_matrix, slices)
    if algorithm == "percentiles":    
        opt_matrix_classified = classification_percentile(opt_matrix, slices)
    if verbose == True:
        print(opt_matrix_classified,"\n")
    
    if surface_plot == True:
        double_parameter_classes_graph(opt_matrix_classified, metric)
        #double_parameter_optimization_graph(opt_matrix_classified, metric)

    rows = len(opt_matrix_classified)
    cols = len(opt_matrix_classified.columns)

    stability_matrix = opt_matrix_classified.copy()
    stability_matrix_lime = pd.DataFrame(columns = opt_matrix_classified.columns, 
                                         index = opt_matrix_classified.index)
    stable_coordinates = []
    fitness_values = []
    opt1_values = []
    opt2_values = []

    for row in range(rows):
        for col in range(cols):
            #print(row, col, opt_matrix_classified.values[row,col])
            stability_matrix.iloc[row,col] = evaluate_point(opt_matrix_classified.values,row,col,inner_tolerance)
            if stability_matrix.iloc[row,col] == 8:
                stable_coordinates.append((row, col))
                fitness_values.append(opt_matrix.iloc[stable_coordinates[-1]])
                opt1_values.append(opt_matrix.index[stable_coordinates[-1][0]])
                opt2_values.append(opt_matrix.index[stable_coordinates[-1][1]])
                stability_matrix_lime.iloc[row,col] = opt_matrix_classified.iloc[row,col]
            #else:
             #   stability_matrix_lime.iloc[row,col] = 0#np.nan
     
    if verbose == True:
        print(stability_matrix_lime,"\n")
    print(X_graph(stability_matrix))
    
    #*********
    #opt_matrix_classified_lime = stability_matrix[stability_matrix == 8]
    #print(stability_matrix)
    double_parameter_classes_graph_stability_points(opt_matrix_classified, stability_matrix_lime, metric)
    #double_parameter_classes_graph(stability_matrix_lime, metric)
    #*********
    
    inner_results = pd.DataFrame(stable_coordinates, columns = ["row","col"])
    inner_results.index = inner_results.index + 1
    inner_results["opt1"] = opt1_values
    inner_results["opt2"] = opt2_values
    inner_results["fitness"] = fitness_values
    
    inner_results = inner_results.sort_values("fitness", ascending = False)

    return opt_matrix, opt_matrix_classified, stability_matrix, inner_results

def multiple_fitness_stability_screener(list_of_stability_matrix, metrics, slices, tolerance, 
                                        algorithm, surface_plot, verbose):
        """
        Function to obtain stable points on multiple fitness 3D surface

        INPUTS
        matrix: optimization results from a double parameter opt
        metric: fitness to calculate z axis
        slices: number of samples from lowest to highest value into the optimization results
        tolerance: percentage difference between class of each point and class of neighboring points
        algorithm: type of algorithm to generate classes 
                   "normalizer": division into "slices" classes the interval between maximum and minimum value 
                                 of the optimization results
                   "percentiles": division into "slices" classes percentiles values
                                 of the optimization results
        surface_plot: plotting of 3D surface
                      True: will plot 3D surface
                      False: won't plot 3D surface

        OUTPUTS
        last_results: coordinates that pass the robustness tests

        """

        stability_matrix_list = []

        for metric in metrics:
            print("\n Metric Analysis on:", metric, "\n")
            stability_results = single_opt_stable_points(list_of_stability_matrix, metric, slices, tolerance,
                                                         "normalizer", surface_plot,verbose)[2]
            stability_matrix_list.append(boolean_matrix(stability_results))
            #print(stability_matrix_list[-1])
            
        i = 0
        for el in stability_matrix_list:
            if i == 0:
                final_result = el
            else:
                final_result = final_result * el
            i += 1

        print("")
        print("Multi Metrics Results:\n")
        print(X_graph_from_boolean(final_result))  
        
        rows = len(final_result)
        cols = len(final_result.columns)

        stable_coordinates = []
        opt1_values = []
        opt2_values = []

        for row in range(rows):
            for col in range(cols):
                if final_result.iloc[row,col] == 1:
                    stable_coordinates.append((row, col))
                    opt1_values.append(final_result.index[stable_coordinates[-1][0]])
                    opt2_values.append(final_result.index[stable_coordinates[-1][1]])
        
        last_results = pd.DataFrame(stable_coordinates, columns = ["row","col"])
        last_results.index = last_results.index + 1
        last_results["opt1"] = opt1_values
        last_results["opt2"] = opt2_values

        return last_results
    
def double_parameter_classes_graph_stability_points(classes_matrix, opt_matrix_lime, optimization_fitness):
    
    import plotly.graph_objs as go

    surface1 = go.Surface(y = classes_matrix.index,
                          x = classes_matrix.columns,
                          z = classes_matrix.values,
                          colorscale = 'RdBu')
    
    surface2 = go.Surface(y = opt_matrix_lime.index,
                          x = opt_matrix_lime.columns,
                          z = opt_matrix_lime.values + 0.01,
                          colorscale = 'greens', showscale = False)

    data = [surface1,surface2]
    
    layout = go.Layout(title = 'Classes 3D Optimization Report - by Gandalf Project R&D',
                       scene = dict(xaxis = dict(title = 'Opt2'),
                                    yaxis = dict(title = 'Opt1'),
                                    zaxis = dict(title = optimization_fitness)),
                       autosize = False,
                       width = 800,
                       height = 600,
                       margin = dict(l = 0, r = 0, b = 50, t = 50))  
        
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)
    
    return

# OPTIMIZATION FUNCTIONS ****************************************************************************************************end
