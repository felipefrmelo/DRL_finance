import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from utils import fill_missing_values
from process_data import FeatureExtractor


MAX_NUM_SHARES = 100000
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

# action constant
BUY = 0
SELL = 1
HOLD = 2

# position constant
LONG = 0
SHORT = 1
FLAT = 2


class StockTradingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=5, show_trade=True):
        super(StockTradingEnv, self).__init__()
        self.show_trade = show_trade
        self.process_data(df)
        self.actions = ["BUY", "SELL", "HOLD"]
        self.window_size = window_size
        self.fee = 0.0005
        self.reward = 0
        self.shape = (self.window_size, len(self.feature_list)+4)
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float16)

    def process_data(self, df):
        # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features
        extractor = FeatureExtractor(df)
        self.df = extractor.add_features(normalized=True)

        # selected manual fetuares
        self.closingPrices = self.df['close'].values
        self.closingPrices[env.closingPrices==0] = self.closingPrices.mean()
        self.feature_list = [
            'ret_day',
            'bnds_value',
            'mean_value',
            'mean_value_41',
            'mean_value_66',
            'momentum_value',
            'rsi_value']
        self.df = fill_missing_values(self.df)  # drops Nan rows
        self.df = self.df[self.feature_list].values

    def step(self, action):

        if self.done:
            return self.state, self.reward, self.done, {}
        self.reward = 0

        # action comes from the agent
        # 0 buy, 1 sell, 2 hold
        # single position can be opened per trade
        # valid action sequence would be
        # LONG : buy - hold - hold - sell
        # SHORT : sell - hold - hold - buy
        # invalid action sequence is just considered hold
        # (e.g.) "buy - buy" would be considred "buy - hold"
        self.action = HOLD  # hold
        if action == BUY:  # buy
            if self.position == FLAT:  # if previous position was flat
                self.position = LONG  # update position to long
                self.action = BUY  # record action as buy
                self.entry_price = self.closingPrice  # maintain entry price
            elif self.position == SHORT:  # if previous position was short
                self.position = FLAT  # update position to flat
                self.action = BUY  # record action as buy
                self.exit_price = self.closingPrice
                self.reward += ((self.entry_price - self.exit_price) /
                                self.exit_price + 1)*(1-self.fee)**2 - 1  # calculate reward
                # evaluate cumulative return in krw-won
                self.krw_balance = self.krw_balance * (1.0 + self.reward)
                self.entry_price = 0  # clear entry price
                self.n_short += 1  # record number of short
        elif action == SELL:  # vice versa for short trade
            if self.position == FLAT:
                self.position = SHORT
                self.action = SELL
                self.entry_price = self.closingPrice
            elif self.position == LONG:
                self.position = FLAT
                self.action = SELL
                self.exit_price = self.closingPrice
                self.reward += ((self.exit_price - self.entry_price) /
                                self.entry_price + 1)*(1-self.fee)**2 - 1
                self.krw_balance = self.krw_balance * (1.0 + self.reward)
                self.entry_price = 0
                self.n_long += 1

        # [coin + krw_won] total value evaluated in krw won
        if(self.position == LONG):
            temp_reward = ((self.closingPrice - self.entry_price) /
                           self.entry_price + 1)*(1-self.fee)**2 - 1
            new_portfolio = self.krw_balance * (1.0 + temp_reward)
        elif(self.position == SHORT):
            temp_reward = ((self.entry_price - self.closingPrice) /
                           self.closingPrice + 1)*(1-self.fee)**2 - 1
            new_portfolio = self.krw_balance * (1.0 + temp_reward)
        else:
            temp_reward = 0
            new_portfolio = self.krw_balance

        self.portfolio = new_portfolio
        self.current_tick += 1
        if(self.show_trade and self.current_tick % 100 == 0):
            print(
                "Tick: {0}/ Portfolio (krw-won): {1}".format(self.current_tick, self.portfolio))
            print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
        self.history.append((self.action, self.current_tick,
                             self.closingPrice, self.portfolio, self.reward))
        self.updateState()
        if (self.current_tick > (self.df.shape[0]) - self.window_size-1):
            self.done = True
            self.reward = self.get_profit()  # return reward at end of the game
        return self.state, self.reward, self.done, {'portfolio': np.array([self.portfolio]),
                                                    "history": self.history,
                                                    "n_trades": {'long': self.n_long, 'short': self.n_short}}

    def get_profit(self):
        if(self.position == LONG):
            profit = ((self.closingPrice - self.entry_price) /
                      self.entry_price + 1)*(1-self.fee)**2 - 1
        elif(self.position == SHORT):
            profit = ((self.entry_price - self.closingPrice) /
                      self.closingPrice + 1)*(1-self.fee)**2 - 1
        else:
            profit = 0
        return profit

    def reset(self):
        # Reset the state of the environment to an initial state
           # self.current_tick = random.randint(0, self.df.shape[0]-1000)
        # print("start episode ... {0} at {1}" .format(self.rand_episode, self.current_step))
        self.current_tick = random.randint(
            0, len(self.closingPrices) - self.window_size)
        print("start episode ... at step {0}" .format(self.current_tick))
        # positions
        self.n_long = 0
        self.n_short = 0
        # clear internal variables
        self.history = []  # keep buy, sell, hold action history
        # initial balance, u can change it to whatever u like
        self.krw_balance = INITIAL_ACCOUNT_BALANCE
        # (coin * current_price + current_krw_balance) == portfolio
        self.portfolio = float(self.krw_balance)
        self.profit = 0
        self.action = HOLD
        self.position = FLAT
        self.done = False
        self.updateState()  # returns observed_features +  opened position(LONG/SHORT/FLAT) + profit_earned(during opened position)
        return self.state

    def updateState(self):
        def one_hot_encode(x, n_classes):
            return np.eye(n_classes)[x]
        self.closingPrice = float(self.closingPrices[self.current_tick])
        prev_position = self.position
        one_hot_position = one_hot_encode(prev_position, 3)
        profit = self.get_profit()
        # append two
        self.state = np.concatenate(
            (self.df[self.current_tick], one_hot_position, [profit]))
        return self.state

    def render(self, mode='human', close=False):
        pass
