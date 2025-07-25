import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas_ta as ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'timestamp': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['date', 'open', 'high', 'low', 'close', 'volume']).reset_index(drop=True)

    df['gap'] = df['close'] - df['open']
    df['high_percent'] = (df['high'] - df['open']) / df['open']
    df['low_percent'] = (df['low'] - df['open']) / df['open']

    df['ema_9'] = ta.ema(df['close'], length=9)
    df['ema_9_slope'] = df['ema_9'].diff()
    df['ema_21'] = ta.ema(df['close'], length=21)
    df['ema_21_slope'] = df['ema_21'].diff()
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    df['rsi_7'] = ta.rsi(df['close'], length=7)
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['boll'] = bbands['BBM_20_2.0']
    df['boll_ub'] = bbands['BBU_20_2.0']
    df['boll_lb'] = bbands['BBL_20_2.0']
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()

    stoch = ta.stoch(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']
    
    df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
    
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)

    df['zscore_close_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    df['rvol'] = df['volume'] / df['volume'].rolling(20).mean()
    df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    return df


def prepare_features(df: pd.DataFrame, window_size: int, extra_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract, align, and normalize price and feature data.
    Returns: (prices, combined_features)
    """
    # Extract raw signals
    signal_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    signal_features = df[signal_cols].values.astype(np.float32)

    # Normalize signal features
    signal_scaler = StandardScaler()
    signal_features = signal_scaler.fit_transform(signal_features)

    # Handle extra features
    extra_features = []
    for col in extra_cols:
        feature = np.nan_to_num(df[col].values.astype(np.float32), nan=0.0).reshape(-1, 1)
        # Normalize each extra feature individually
        scaler = StandardScaler()
        feature = scaler.fit_transform(feature)
        extra_features.append(feature)

    # Cut off first `window_size` rows to align with env's state shape
    signal_features = signal_features[window_size:]
    aligned_extra = [f[window_size:] for f in extra_features]

    # Stack everything together
    combined = np.hstack([signal_features] + aligned_extra)
    prices = df['Close'].values[window_size:]

    return prices, combined


def normalize_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df