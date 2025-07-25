import os
import pandas as pd
import pandas_ta as ta
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import numpy as np
from stable_baselines3 import PPO

from add_indicators import add_indicators, normalize_features
from trading_env import CustomStocksWithIndicatorsEnv

from evaluation import EvalProfitCallback, ProfitToTensorboardCallback, make_eval_env

# 3. Load data and prepare it
df = pd.read_csv("data/btc_1m.csv", parse_dates=['timestamp'])
df = add_indicators(df)

# 4. Prepare env input for gym-anytrading
df = df.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
df.set_index("Date", inplace=True)


indicators = ['ema_9', 'ema_21', 'macd', 'rsi_7', 
    'ema_9_slope', 'ema_21_slope',
    'gap', 'high_percent', 'low_percent',
    'atr_14', 'boll', 'boll_ub', 'boll_lb', 'vwap']
df = normalize_features(df, indicators)

split_index = int(len(df) * 0.9)
df_train = df.iloc[:split_index].copy()
df_eval = df.iloc[split_index - 60:].copy()  # include window overlap

env = CustomStocksWithIndicatorsEnv(df=df_train, window_size=60, frame_bound=(60, len(df_train) - 1), max_episode_steps=4096, features=indicators)
env.trade_fee_ask_percent = 0.0004
env.trade_fee_bid_percent = 0.0004
env = Monitor(env)

# 6. Set up logging
log_dir = "./ppo_btc_logs/"
os.makedirs(log_dir, exist_ok=True)
logger = configure(log_dir, ["stdout", "tensorboard"])

# 7. Train with PPO (not RecurrentPPO, gym-anytrading doesn't support it by default)
model = PPO(
    "MlpPolicy",
    env,
    n_steps=4096,
    batch_size=128,
    learning_rate=1e-5, 
    gamma=0.999,
    gae_lambda=0.97,
    ent_coef=0.05,
    vf_coef=0.1,
    max_grad_norm=0.9,
    clip_range=0.2,
    tensorboard_log=log_dir,
    verbose=1,
    normalize_advantage=True
)
model.set_logger(logger)


eval_callback = EvalProfitCallback(make_eval_env(df_eval), eval_freq=100_000, n_eval_episodes=1)


# 9. Train model
callback = [eval_callback, ProfitToTensorboardCallback(verbose=0)]
model.learn(total_timesteps=5_000_000, progress_bar=True, callback=callback,)


