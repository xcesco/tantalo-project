import ccxt
import pandas as pd
from sklearn import preprocessing
import pprint
import datetime

print('ccxt version', ccxt.__version__)


# From timestamp to Datetime
def convert_from_ms(value):
    return datetime.datetime.fromtimestamp(value)


exchange = ccxt.binance()
markets = exchange.load_markets()
print(f"numero mercati: {len(markets)}")

symbol = 'BTC/EUR'
filtered = []
for s in exchange.symbols:
    if s.startswith('BTC'):
        filtered.append(s)

# print(filtered)
ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=1000)
if len(ohlcv):
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    # df['datetime']=pd.to_datetime(df['timestamp'],unit='ms')
    df['date_parsed'] = (df['timestamp'] / 1000).apply(convert_from_ms)
    df['avg'] = df['high'] - df['low'] / 2

    x = df.values  # returns a numpy array
 #   min_max_scaler = preprocessing.MinMaxScaler()
 #   x_scaled = min_max_scaler.fit_transform(x)
 #   df = pd.DataFrame(x_scaled)

 #   x = df  # returns a numpy array
 #   min_max_scaler = preprocessing.MinMaxScaler()
 #   x_scaled = min_max_scaler.fit_transform(x)
 #   print(x_scaled)
    # df['norm_close'] = pd.DataFrame(min_max_scaler.fit_transform(x))

    # df['date']=instrument['date_parsed']
    df.index = df['date_parsed']
    df.drop('date_parsed', axis=1, inplace=True)
    df.drop('timestamp', axis=1, inplace=True)

print(df)
df['avg'].plot(kind = 'bar')
