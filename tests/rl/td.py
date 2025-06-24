import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from scipy.stats import norm

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)


# Black-Scholes模型计算函数
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """计算欧式期权的Black-Scholes价格和希腊值"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma  ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    # 计算Gamma和Vega(对call和put相同)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return price, delta, gamma, vega


# 环境类 - 模拟期权对冲环境
class OptionHedgingEnv:
    def __init__(self, S0=100, K=100, T=1.0, r=0.05, sigma=0.2, dt=1 / 252,
                 transaction_cost=0.001, option_type='call'):
        """
        初始化期权对冲环境
        参数:
            S0: 初始标的资产价格
            K: 期权执行价
            T: 期权到期时间(年)
            r: 无风险利率
            sigma: 标的资产波动率
            dt: 时间步长(年)
            transaction_cost: 交易成本比例
            option_type: 期权类型('call'或'put')
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.dt = dt
        self.transaction_cost = transaction_cost
        self.option_type = option_type

        # 重置环境
        self.reset()

    def reset(self):
        """重置环境到初始状态"""
        self.current_S = self.S0
        self.current_t = 0
        self.current_position = 0  # 初始持仓为0
        self.cash = 0  # 初始现金为0
        self.done = False

        # 计算初始期权价格和希腊值
        self.option_price, self.delta, self.gamma, self.vega = black_scholes(
            self.current_S, self.K, self.T - self.current_t, self.r, self.sigma, self.option_type)

        return self._get_state()

    def step(self, action):
        """
        执行一个动作并返回(next_state, reward, done, info)
        动作: 要买入/卖出的股票数量(正为买入，负为卖出)
        """
        if self.done:
            raise ValueError("Episode已经结束，请调用reset()")

        # 1. 执行交易并计算交易成本
        shares_traded = action - self.current_position  # 需要交易的股票数量
        cost = abs(shares_traded) * self.current_S * self.transaction_cost
        self.cash -= shares_traded * self.current_S - cost  # 更新现金账户

        # 2. 更新持仓
        self.current_position = action

        # 3. 模拟下一个时间步的资产价格变化(几何布朗运动)
        self.current_t += self.dt
        z = np.random.normal(0, 1)
        self.current_S *= np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt +
                                 self.sigma * np.sqrt(self.dt) * z)

        # 4. 计算新的期权价格和希腊值
        remaining_time = max(self.T - self.current_t, 0)
        self.option_price, self.delta, self.gamma, self.vega = black_scholes(
            self.current_S, self.K, remaining_time, self.r, self.sigma, self.option_type)

        # 5. 计算投资组合价值变化
        portfolio_value = self.current_position * self.current_S + self.cash
        option_payoff = max(self.current_S - self.K, 0) if self.option_type == 'call' else max(self.K - self.current_S,
                                                                                               0)

        # 6. 检查是否到期
        if self.current_t >= self.T:
            self.done = True
            # 到期时的reward是最终对冲误差(越小越好)
            reward = -abs(portfolio_value - option_payoff)
        else:
            # 未到期时的reward是投资组合价值变化(考虑风险)
            reward = -abs(self.current_position * (self.current_S - self.current_S / (1 + self.r * self.dt)) -
                          (self.option_price - self.option_price / (1 + self.r * self.dt)))

        return self._get_state(), reward, self.done, {}

    def _get_state(self):
        """获取当前状态"""
        return np.array([
            self.current_S / self.S0,  # 标准化价格
            (self.T - self.current_t) / self.T,  # 标准化剩余时间
            self.current_position,  # 当前持仓
            self.delta,  # BS Delta
            self.gamma,  # BS Gamma
            self.vega  # BS Vega
        ], dtype=np.float32)

    def get_bs_delta(self):
        """返回当前Black-Scholes Delta"""
        return self.delta


# Q网络 - 近似Q(s,a)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# SARSA Agent
class SARSAAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99,
                 epsilon=0.1, epsilon_decay=0.999, min_epsilon=0.01):
        """
        初始化SARSA Agent
        参数:
            state_dim: 状态维度
            action_dim: 动作维度(离散动作数量)
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: ε-greedy策略的初始ε值
            epsilon_decay: ε衰减率
            min_epsilon: 最小ε值
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # 初始化Q网络和目标Q网络
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_q_net = QNetwork(state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        # 动作空间(离散持仓水平)
        self.actions = np.linspace(-1.5, 1.5, action_dim)  # 从-1.5到1.5的持仓水平

    def get_action(self, state, greedy=False):
        """根据当前状态选择动作(ε-greedy策略)"""
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_net(state).detach().numpy()[0]

        if greedy or np.random.rand() > self.epsilon:
            return np.argmax(q_values)
        else:
            return np.random.randint(self.action_dim)

    def update_epsilon(self):
        """更新ε值"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update(self, state, action, reward, next_state, next_action, done):
        """
        使用SARSA更新Q网络
        SARSA更新公式: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        # 当前Q值
        current_q = self.q_net(state)[0, action]

        # 下一个Q值(使用目标网络)
        with torch.no_grad():
            next_q = self.target_q_net(next_state)[0, next_action]

        # 计算目标Q值
        target_q = reward + (1 - done) * self.gamma * next_q

        # 计算损失并更新
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_q_net.load_state_dict(self.q_net.state_dict())


# 训练函数
def train_agent(env, agent, episodes=1000, batch_size=32, target_update_freq=10):
    """训练SARSA Agent"""
    episode_rewards = []
    episode_hedging_errors = []
    bs_hedging_errors = []  # 用于比较BS Delta对冲

    for episode in range(episodes):
        state = env.reset()
        action = agent.get_action(state)
        done = False
        total_reward = 0
        portfolio_values = []
        bs_portfolio_values = []

        # 初始持仓和现金
        bs_position = env.get_bs_delta()
        bs_cash = env.option_price - bs_position * env.current_S

        while not done:
            # 执行动作
            next_state, reward, done, _ = env.step(agent.actions[action])
            total_reward += reward

            # 选择下一个动作
            next_action = agent.get_action(next_state)

            # 更新Q网络
            loss = agent.update(state, action, reward, next_state, next_action, done)

            # 记录投资组合价值
            portfolio_value = env.current_position * env.current_S + env.cash_1
            portfolio_values.append(portfolio_value)

            # 记录BS Delta对冲的投资组合价值
            bs_position = env.get_bs_delta()
            bs_cash = bs_cash - (bs_position - env.current_position) * env.current_S  # 更新现金账户
            bs_portfolio_value = bs_position * env.current_S + bs_cash
            bs_portfolio_values.append(bs_portfolio_value)

            # 更新状态和动作
            state = next_state
            action = next_action

        # 计算对冲误差
        option_payoff = max(env.current_S - env.K, 0) if env.option_type == 'call' else max(env.K - env.current_S, 0)
        hedging_error = abs(portfolio_values[-1] - option_payoff)
        bs_hedging_error = abs(bs_portfolio_values[-1] - option_payoff)

        # 记录结果
        episode_rewards.append(total_reward)
        episode_hedging_errors.append(hedging_error)
        bs_hedging_errors.append(bs_hedging_error)

        # 更新ε和目标网络
        agent.update_epsilon()
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # 打印进度
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}, "
                  f"Hedging Error: {hedging_error:.2f}, BS Error: {bs_hedging_error:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")

    return episode_rewards, episode_hedging_errors, bs_hedging_errors


# 主函数
def main():
    # 创建环境和Agent
    env = OptionHedgingEnv(S0=100, K=100, T=0.5, r=0.05, sigma=0.2, dt=1 / 252,
                           transaction_cost=0.001, option_type='call')

    state_dim = len(env._get_state())
    action_dim = 21  # 离散动作数量(从-1.5到1.5的持仓水平)

    agent = SARSAAgent(state_dim, action_dim, learning_rate=0.001, gamma=0.99,
                       epsilon=0.5, epsilon_decay=0.995, min_epsilon=0.01)

    # 训练Agent
    episodes = 1000
    episode_rewards, episode_hedging_errors, bs_hedging_errors = train_agent(
        env, agent, episodes=episodes)

    # 绘制结果
    plt.figure(figsize=(12, 8))

    # 奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    # 对冲误差曲线
    plt.subplot(2, 2, 2)
    plt.plot(episode_hedging_errors, label='RL Agent')
    plt.plot(bs_hedging_errors, label='BS Delta')
    plt.title("Hedging Errors at Maturity")
    plt.xlabel("Episode")
    plt.ylabel("Absolute Hedging Error")
    plt.legend()

    # 滑动平均对冲误差
    window_size = 50
    rl_ma = np.convolve(episode_hedging_errors, np.ones(window_size) / window_size, mode='valid')
    bs_ma = np.convolve(bs_hedging_errors, np.ones(window_size) / window_size, mode='valid')

    plt.subplot(2, 2, 3)
    plt.plot(rl_ma, label='RL Agent (MA)')
    plt.plot(bs_ma, label='BS Delta (MA)')
    plt.title(f"Moving Average Hedging Errors (Window={window_size})")
    plt.xlabel("Episode")
    plt.ylabel("MA Hedging Error")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 测试训练好的Agent
    test_episodes = 100
    test_hedging_errors = []
    test_bs_errors = []

    for _ in range(test_episodes):
        state = env.reset()
        done = False
        portfolio_values = []
        bs_portfolio_values = []

        # 初始持仓和现金
        bs_position = env.get_bs_delta()
        bs_cash = env.option_price - bs_position * env.current_S

        while not done:
            # 使用训练好的策略(贪婪)
            action = agent.get_action(state, greedy=True)
            state, _, done, _ = env.step(agent.actions[action])

            # 记录投资组合价值
            portfolio_value = env.current_position * env.current_S + env.cash
            portfolio_values.append(portfolio_value)

            # 记录BS Delta对冲的投资组合价值
            bs_position = env.get_bs_delta()
            bs_cash = bs_cash - (bs_position - env.current_position) * env.current_S  # 更新现金账户
            bs_portfolio_value = bs_position * env.current_S + bs_cash
            bs_portfolio_values.append(bs_portfolio_value)

        # 计算对冲误差
        option_payoff = max(env.current_S - env.K, 0) if env.option_type == 'call' else max(env.K - env.current_S, 0)
        hedging_error = abs(portfolio_values[-1] - option_payoff)
        bs_hedging_error = abs(bs_portfolio_values[-1] - option_payoff)

        test_hedging_errors.append(hedging_error)
        test_bs_errors.append(bs_hedging_error)

    print("\nTest Results:")
    print(f"RL Agent - Mean Hedging Error: {np.mean(test_hedging_errors):.4f}, Std: {np.std(test_hedging_errors):.4f}")
    print(f"BS Delta - Mean Hedging Error: {np.mean(test_bs_errors):.4f}, Std: {np.std(test_bs_errors):.4f}")


if __name__ == "__main__":
    main()