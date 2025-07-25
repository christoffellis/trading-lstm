import numpy as np
from gym_anytrading.envs import StocksEnv, Actions, Positions
import numpy as np
from add_indicators import prepare_features

# 5. Create gym-anytrading environment
class CustomStocksWithIndicatorsEnv(StocksEnv):
    def __init__(self, df, window_size=60, frame_bound=(60, 1000), max_episode_steps=4096, features=[]):
        self.df = df
        self.extra_features = features
        self.spread = 50

        self.max_episode_steps = max_episode_steps
        self.current_episode_steps = 0

        super().__init__(df=df, window_size=window_size, frame_bound=frame_bound)

    def _process_data(self):
        return prepare_features(self.df, self.window_size, self.extra_features)


    def _calculate_reward(self, action):
        if self._current_tick >= len(self.prices):
            return 0.0
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        step_reward = 0.0
        trade = False
        
        if self._position == Positions.Long:
            possible_profit = (self.prices[self._last_trade_tick] - self.prices[self._current_tick]) / self.prices[self._last_trade_tick]
        else: 
            possible_profit = (self.prices[self._current_tick] - self.prices[self._last_trade_tick]) / self.prices[self._last_trade_tick]

        fee_pct = 0.0004

        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade:

            # Percent return
            if self._position == Positions.Long:
                pct_return = (current_price - last_trade_price) / last_trade_price
            else:
                pct_return = (last_trade_price - current_price) / last_trade_price

            step_reward += pct_return - fee_pct

            if step_reward < 1:
                step_reward /= 2
            
            step_reward *= 1000  # Scale reward
        # else:
        #     # Small holding reward based on drift
        #     if self._position == Positions.Long:
        #         drift = (current_price - last_trade_price) / last_trade_price
        #     elif self._position == Positions.Short:
        #         drift = (last_trade_price - current_price) / last_trade_price
        #     else:
        #         drift = 0.0
        #     step_reward += 10 * drift

        #     # Small context-aware penalty for doing nothing
        #     atr = self.df['atr_14'].iloc[self._current_tick]
        #     step_reward -= 0.00005 * atr

        step_reward = np.clip(step_reward, -1, 1) 
        
        # print(f"Action: {action}, Current Price: {current_price}, Last Trade Price: {last_trade_price}, Step Reward: {step_reward}")

        return step_reward
    
    def _calculate_floating_pnl(self) -> float:
        if self._current_tick >= len(self.prices):
            return 0.0
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]

        if self._position == Positions.Long:
            return current_price - last_trade_price
        elif self._position == Positions.Short:
            return last_trade_price - current_price
        return 0.0


    def _get_observation(self):
        obs = super()._get_observation()  # Shape: (window_size, num_features)

        # Prepare new features
        current_position = float(self._position.value)  # Assuming it's an enum
        floating_pnl = self._calculate_floating_pnl()

        # Create padding for all but the last frame
        extra_features = np.zeros((self.window_size, 2), dtype=np.float32)
        # add current pos, floating pnl, and time till
        extra_features[-1] = [current_position, floating_pnl]

        # Concatenate original and extra features
        full_obs = np.concatenate([obs, extra_features], axis=1)  # Shape: (window_size, num_features + 2)
        return full_obs



    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)

        self.current_episode_steps += 1
        if self._total_profit < 0.7 or self._total_profit > 1.5:
            done = True
            info['EarlyStop.profit'] = self._total_profit

        if done:
            self.current_episode_steps = 0
            info['episode'] = {
                'r': self._total_reward,
                'p': self._total_profit,
                'l': self._last_trade_tick - self._start_tick,
                't': self._current_tick - self._start_tick
            }

            info["df"] = self.df
            info["start_tick"] = self.frame_bound[0]
            info["end_tick"] = self._current_tick

            if self.render_mode == 'human':
                self._render_frame()

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Seed the environment's RNG if needed
        if seed is not None:
            self.action_space.seed(seed)
            np.random.seed(seed)
        
        # 🟡 Randomize start tick within frame_bound
        start_range = self.frame_bound[1] - self.max_episode_steps 
        self._start_tick = np.random.randint(self.frame_bound[0], start_range)
        self._end_tick = self._start_tick + self.max_episode_steps
        
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.0  # unit starting capital
        self._truncated = False
        self._first_rendering = True
        self.history = {}

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info


    # def _get_observation(self):
    #     start = self._start
    #     step = self._current_step
    #     return self.signal_features[(start + step - self.window_size):(start + step)]
