import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

# Setup
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1m'
since = exchange.parse8601('2023-01-01T00:00:00Z')
now = exchange.milliseconds()

# Collect data
all_ohlcv = []
batch_ms = 1000 * 60 * 1000  # 1000 minutes in ms (~16.6 hrs)

while since < now:
    print(f"Fetching data since {exchange.iso8601(since)}")
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        since = ohlcv[-1][0] + 60_000  # advance by 1 min
        time.sleep(0.5)  # rate limit
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)

# Convert to DataFrame
df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

print(df.head())
print(f"Total rows: {len(df)}")
df.to_csv('BTCUSDT_1m_2024_to_2025.csv', index=False)
