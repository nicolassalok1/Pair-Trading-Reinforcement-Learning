import pandas as pd
import numpy as np
import MAIN.Basics as basics
import MAIN.Reinforcement as RL
import tensorflow.compat.v1 as tf
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from UTIL import FileIO
from STRATEGY.Cointegration import EGCointegration

tf.disable_v2_behavior()


# Paths
repo_root   = Path(__file__).resolve().parent.parent
config_path = repo_root / "CONFIG" / "config_train.yml"
price_dir   = repo_root / "STATICS" / "PRICE"
ticker_x    = "BTC"
ticker_y    = "USD"


def _ensure_price_pair(path_a: Path, path_b: Path, base_a: float, base_b: float,
                       vol_a: float, vol_b: float, corr: float, n_obs: int = 8000):
    """
    If either price file is missing, generate a correlated pair so the example can run end-to-end.
    """
    if path_a.exists() and path_b.exists():
        return path_a, path_b

    rng = np.random.default_rng(seed=123)
    cov = np.array([[1.0, corr], [corr, 1.0]])
    L = np.linalg.cholesky(cov)
    shocks = rng.standard_normal(size=(n_obs, 2)) @ L.T

    dt = 1 / 1440  # minute-ish step for illustration
    mu = 0.02
    ret_a = np.exp((mu - 0.5 * vol_a ** 2) * dt + vol_a * shocks[:, 0] * np.sqrt(dt))
    ret_b = np.exp((mu - 0.5 * vol_b ** 2) * dt + vol_b * shocks[:, 1] * np.sqrt(dt))
    prices_a = base_a * np.cumprod(ret_a)
    prices_b = base_b * np.cumprod(ret_b)

    dates = pd.date_range("2020-01-01", periods=n_obs, freq="T")
    df_a = pd.DataFrame({"date": dates.strftime("%Y-%m-%d %H:%M:%S"), "close": prices_a})
    df_b = pd.DataFrame({"date": dates.strftime("%Y-%m-%d %H:%M:%S"), "close": prices_b})

    path_a.parent.mkdir(parents=True, exist_ok=True)
    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)
    print(f"[Info] Created synthetic price files:\n  {path_a}\n  {path_b}")
    return path_a, path_b

# Read config
config_train = FileIO.read_yaml(str(config_path))

# Ensure data is available; synthesize if missing to keep the example runnable
path_x, path_y = _ensure_price_pair(
    price_dir / f"{ticker_x}.csv",
    price_dir / f"{ticker_y}.csv",
    base_a=30000, base_b=1.3,
    vol_a=0.6, vol_b=0.25,
    corr=0.7,
    n_obs=8000,
)

# Read prices
x = pd.read_csv(path_x)
y = pd.read_csv(path_y)
x, y = EGCointegration.clean_data(x, y, 'date', 'close')

# Separate training and testing sets
train_pct = 0.7
train_len = round(len(x) * 0.7)
idx_train = list(range(0, train_len))
idx_test  = list(range(train_len, len(x)))
EG_Train = EGCointegration(x.iloc[idx_train, :], y.iloc[idx_train, :], 'date', 'close')
EG_Test  = EGCointegration(x.iloc[idx_test,  :], y.iloc[idx_test,  :], 'date', 'close')

# Create action space
n_hist    = list(np.arange(60, 601, 60))
n_forward = list(np.arange(120, 1201, 120))
trade_th  = list(np.arange(1,  5.1, 1))
stop_loss = list(np.arange(1,  2.1, 0.5))
cl        = list(np.arange(0.05,  0.11, 0.05))
actions   = {'n_hist':    n_hist,
             'n_forward': n_forward,
             'trade_th':  trade_th,
             'stop_loss': stop_loss,
             'cl':        cl}
n_action  = int(np.prod([len(actions[key]) for key in actions.keys()]))

# Create state space
transaction_cost = [0.001]
states  = {'transaction_cost': transaction_cost}
n_state = len(states)

# Assign state and action spaces to config
config_train['StateSpaceState'] = states
config_train['ActionSpaceAction'] = actions

# Create and build network
one_hot  = {'one_hot': {'func_name':  'one_hot',
                        'input_arg':  'indices',
                         'layer_para': {'indices': None,
                                        'depth': n_state}}}
output_layer = {'final': {'func_name':  'fully_connected',
                          'input_arg':  'inputs',
                          'layer_para': {'inputs': None,
                                         'num_outputs': n_action,
                                         'biases_initializer': None,
                                         'activation_fn': tf.nn.relu,
                                         'weights_initializer': tf.ones_initializer()}}}

state_in = tf.placeholder(shape=[1], dtype=tf.int32)

N = basics.Network(state_in)
N.build_layers(one_hot)
N.add_layer_duplicates(output_layer, 1)

# Create learning object and perform training
RL_Train = RL.ContextualBandit(N, config_train, EG_Train)

sess = tf.Session()
RL_Train.process(sess, save=False, restore=False)

# Extract training results
action = RL_Train.recorder.record['NETWORK_ACTION']
reward = RL_Train.recorder.record['ENGINE_REWARD']
print(np.mean(reward))

df1 = pd.DataFrame()
df1['action'] = action
df1['reward'] = reward
mean_reward = df1.groupby('action').mean()
sns.distplot(mean_reward)

# Test by trading continuously
[opt_action] = sess.run([RL_Train.output], feed_dict=RL_Train.feed_dict)
opt_action = np.argmax(opt_action)
action_dict = RL_Train.action_space.convert(opt_action, 'index_to_dict')
indices = range(601, len(EG_Test.x) - 1200)

pnl = pd.DataFrame()
pnl['Time'] = EG_Test.timestamp
pnl['Trade_Profit'] = 0
pnl['Cost'] = 0
pnl['N_Trade'] = 0

import warnings
warnings.filterwarnings('ignore')
for i in indices:
    if i % 100 == 0:
        print(i)
    EG_Test.process(index=i, transaction_cost=0.001, **action_dict)
    trade_record = EG_Test.record
    if (trade_record is not None) and (len(trade_record) > 0):
        print('value at {}'.format(i))
        trade_record = pd.DataFrame(trade_record)
        trade_cost   = trade_record.groupby('trade_time')['trade_cost'].sum()
        close_cost   = trade_record.groupby('close_time')['close_cost'].sum()
        profit       = trade_record.groupby('close_time')['profit'].sum()
        open_pos     = trade_record.groupby('trade_time')['long_short'].sum()
        close_pos    = trade_record.groupby('close_time')['long_short'].sum() * -1

        pnl['Cost'].loc[pnl['Time'].isin(trade_cost.index)] += trade_cost.values
        pnl['Cost'].loc[pnl['Time'].isin(close_cost.index)] += close_cost.values
        pnl['Trade_Profit'].loc[pnl['Time'].isin(close_cost.index)] += profit.values
        pnl['N_Trade'].loc[pnl['Time'].isin(trade_cost.index)] += open_pos.values
        pnl['N_Trade'].loc[pnl['Time'].isin(close_cost.index)] += close_pos.values

warnings.filterwarnings(action='once')

# Plot the testing result
pnl['PnL'] = (pnl['Trade_Profit'] - pnl['Cost']).cumsum()
plt.plot(pnl['PnL'])
plt.plot(pnl['N_Trade'])
plt.plot(pnl['Time'], pnl['PnL'])

plt.plot(pnl['Time'], pnl['N_Trade'])

sess.close()

