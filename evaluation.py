import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from gym_anytrading.envs import StocksEnv, Actions, Positions
from trading_env import CustomStocksWithIndicatorsEnv
from stable_baselines3.common.monitor import Monitor

def make_eval_env(df_eval, features=[]):
    env = CustomStocksWithIndicatorsEnv(
        df=df_eval,
        window_size=60,
        frame_bound=(60, len(df_eval)-1),
        max_episode_steps=len(df_eval) - 60 * 12,
        features=features
    )
    env.trade_fee_ask_percent = 0.00042
    env.trade_fee_bid_percent = 0.00042
    return Monitor(env)

import numpy as np

def evaluate_with_profit(model, env, n_eval_episodes=1):
    profits = []
    episode_rewards = []
    episode_lengths = []
    max_profits = []
    profit_factors = []
    win_ratios = []
    sharpe_ratios = []
    max_drawdowns = []
    trade_counts = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        total_reward = 0.0
        length = 0
        last_profit = None
        max_profit = 0.0

        returns = []
        gains = 0.001
        gains_count = 0
        losses = 0.001
        losses_count = 0

        profit_curve = []

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            length += 1

            current_profit = info.get("total_profit", None)
            if current_profit is not None:
                profit_curve.append(current_profit)

                if last_profit is not None:
                    delta = current_profit - last_profit
                    returns.append(delta)

                    if delta > 0:
                        gains += delta
                        gains_count += 1
                    elif delta < 0:
                        losses += -delta
                        losses_count += 1

                last_profit = current_profit
                max_profit = max(max_profit, current_profit)

        # Final metrics
        final_profit = round(last_profit, 2) if last_profit is not None else 0.0
        profits.append(final_profit)
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        max_profits.append(max_profit)

        profit_factors.append(gains / losses)
        total_trades = gains_count + losses_count
        win_ratios.append(gains_count / total_trades if total_trades > 0 else 0.0)
        trade_counts.append(total_trades)

        returns = np.array(returns)
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns) + 1e-8
            sharpe = mean_return / std_return
        else:
            sharpe = 0.0
        sharpe_ratios.append(sharpe)

        if profit_curve:
            profit_array = np.array(profit_curve)
            running_max = np.maximum.accumulate(profit_array)
            drawdowns = (running_max - profit_array) / (running_max + 1e-8)
            max_drawdowns.append(np.max(drawdowns))
        else:
            max_drawdowns.append(0.0)

    return (
        profits,
        episode_rewards,
        episode_lengths,
        max_profits,
        profit_factors,
        win_ratios,
        sharpe_ratios,
        max_drawdowns,
        np.mean(trade_counts),
    )

class EvalProfitCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10_000, n_eval_episodes=1, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            (
                profits,
                rewards,
                lengths,
                max_profits,
                profit_factors,
                win_ratios,
                sharpe_ratios,
                max_drawdowns,
                trade_count
            ) = evaluate_with_profit(self.model, self.eval_env, self.n_eval_episodes)

            mean_profit = np.mean(profits)
            mean_reward = np.mean(rewards)
            mean_length = np.mean(lengths)
            mean_max_profit = np.mean(max_profits)
            mean_profit_factor = np.mean(profit_factors)
            mean_win_ratio = np.mean(win_ratios)
            mean_sharpe = np.mean(sharpe_ratios)
            mean_max_drawdown = np.mean(max_drawdowns)

            # Logging to TensorBoard
            self.logger.record("eval/profit/total", round(mean_profit, 2))
            self.logger.record("eval/profit/max", mean_max_profit)
            self.logger.record("eval/profit/profit_factor", mean_profit_factor)
            self.logger.record("eval/profit/win_ratio", mean_win_ratio)
            self.logger.record("eval/profit/sharpe_ratio", mean_sharpe)
            self.logger.record("eval/profit/max_drawdown", mean_max_drawdown)
            self.logger.record("eval/episode_reward", mean_reward)
            self.logger.record("eval/episode_length", mean_length)
            self.logger.record("eval/trade_count", trade_count)

            if self.verbose:
                print(
                    f"[Eval] Profit: {mean_profit:.4f}, "
                    f"Reward: {mean_reward:.4f}, "
                    f"Length: {mean_length:.0f}, "
                    f"Sharpe: {mean_sharpe:.4f}, "
                )

        return True
    
class ProfitToTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "total_profit" in info:
                total_profit = round(info["total_profit"], 4)

                # --- Buy and Hold Calculation ---
                if "df" in info:
                    df = info["df"]
                    start_index = info.get("start_tick", 0)
                    end_index = info.get("end_tick", len(df) - 1)

                    start_price = df.iloc[start_index]["Close"]
                    end_price = df.iloc[end_index]["Close"]

                    buy_and_hold_profit = round(end_price / start_price, 4)
                    delta = round(total_profit - buy_and_hold_profit, 4)

                    self.logger.record("rollout/total_profit", total_profit)
                    self.logger.record("rollout/buy_and_hold", buy_and_hold_profit)
                    self.logger.record("rollout/profit_delta", delta)

                    if self.verbose:
                        print(
                            f"[Episode {self.episode_count}] Total Profit: {total_profit:.4f} | "
                            f"Buy & Hold: {buy_and_hold_profit:.4f} | Delta: {delta:.4f}"
                        )
                else:
                    # If df not in info, skip buy-and-hold
                    self.logger.record("rollout/total_profit", total_profit)

                self.episode_count += 1

        return True

