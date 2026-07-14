import numpy as np
import matplotlib.pyplot as plt

class NonStationaryBandit:
    def __init__(self, k=10):
        self.k = k
        self.q_true = np.zeros(k) # all q*(a) start equal
        
    def step(self):
        self.q_true += np.random.normal(0, 0.01, self.k)
        return self.q_true
    
    def get_reward(self, action):
        return np.random.normal(self.q_true[action], 1.0)

class Agent:
    def __init__(self, k):
        self.k = k
        self.q_values = np.zeros(k)

class EpsilonGreedyAgent(Agent):
    def __init__(self, k, epsilon, alpha=0.1):
        super().__init__(k)
        self.epsilon = epsilon
        self.alpha = alpha
        
    def choose_action(self):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.k) # exploration 
        else:
            best = np.flatnonzero(self.q_values == self.q_values.max())
            action = np.random.choice(best) # exploitation 
        return action
    
    def update(self, action, reward):
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

def run_single_experiment(epsilon, steps=200000, k=10, alpha=0.1):
    env = NonStationaryBandit(k=k)
    agent = EpsilonGreedyAgent(k=k, epsilon=epsilon, alpha=alpha)
    rewards = np.zeros(steps)

    for t in range(steps):
        action = agent.choose_action()
        reward = env.get_reward(action)
        agent.update(action, reward)
        rewards[t] = reward
        env.step()
    return rewards

def evaluate_epsilon(epsilon, runs=20, steps=200000, k=10, alpha=0.1):
    avg_rewards = []
    for _ in range(runs):
        rewards = run_single_experiment(epsilon=epsilon, steps=steps, k=k, alpha=alpha)
        avg_last_100k = np.mean(rewards[-100000:])
        avg_rewards.append(avg_last_100k)
    return np.mean(avg_rewards)

class UCBAgent(Agent): 
    def __init__(self, k, c, alpha=0.1):
        super().__init__(k)
        self.c = c
        self.alpha = alpha
        self.action_counts = np.zeros(k)
        self.t = 0
        
    def choose_action(self):
        self.t += 1
        for action in range(self.k):
            if self.action_counts[action] == 0:
                return action
            
        ucb_values = self.q_values + self.c * np.sqrt(np.log(self.t) / self.action_counts)
        best_actions = np.flatnonzero(ucb_values == ucb_values.max())
        return np.random.choice(best_actions)
    
    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

def run_single_ucb_experiment(c, steps=200000, k=10, alpha=0.1):
    env = NonStationaryBandit(k=k)
    agent = UCBAgent(k=k, c=c, alpha=alpha)
    rewards = np.zeros(steps)

    for t in range(steps):
        action = agent.choose_action()
        reward = env.get_reward(action)
        agent.update(action, reward)
        rewards[t] = reward
        env.step()
    return rewards

def evaluate_ucb(c, runs=20, steps=200000, k=10, alpha=0.1):
    avg_rewards = []
    for _ in range(runs):
        rewards = run_single_ucb_experiment(c=c, steps=steps, k=k, alpha=alpha)
        avg_last_100k = np.mean(rewards[-100000:])
        avg_rewards.append(avg_last_100k)
    return np.mean(avg_rewards)

class GradientBanditAgent(Agent):
    def __init__(self, k, alpha, alpha_r=0.1):
        super().__init__(k)
        self.alpha = alpha
        self.alpha_r = alpha_r  
        self.preferences = np.zeros(k)
        self.avg_reward = 0.0

    def get_action_probabilities(self):
        shifted = self.preferences - np.max(self.preferences)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)

    def choose_action(self):
        probabilities = self.get_action_probabilities()
        return np.random.choice(self.k, p=probabilities)

    def update(self, action, reward):
        probabilities = self.get_action_probabilities()
        baseline = self.avg_reward
    
        for a in range(self.k):
            if a == action:
                self.preferences[a] += self.alpha * (reward - baseline) * (1.0 - probabilities[a])
            else:
                self.preferences[a] -= self.alpha * (reward - baseline) * probabilities[a]
        self.avg_reward += self.alpha_r * (reward - self.avg_reward)

def run_single_gradient_experiment(alpha, steps=200000, k=10):
    env = NonStationaryBandit(k=k)
    agent = GradientBanditAgent(k=k, alpha=alpha)
    rewards = np.zeros(steps)

    for t in range(steps):
        action = agent.choose_action()
        reward = env.get_reward(action)
        agent.update(action, reward)
        rewards[t] = reward
        env.step()
    return rewards

def evaluate_gradient(alpha, runs=20, steps=200000, k=10):
    avg_rewards = []
    for _ in range(runs):
        rewards = run_single_gradient_experiment(alpha=alpha, steps=steps, k=k)
        avg_last_100k = np.mean(rewards[-100000:])
        avg_rewards.append(avg_last_100k)
    return np.mean(avg_rewards)

class OptimisticGreedyAgent(Agent):
    def __init__(self, k, initial_value=5.0, alpha=0.1):
        super().__init__(k)
        self.alpha = alpha
        self.q_values = np.ones(k) * initial_value

    def choose_action(self):
        best = np.flatnonzero(self.q_values == self.q_values.max())
        return np.random.choice(best)

    def update(self, action, reward):
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

def run_single_optimistic_experiment(initial_value, steps=200000, k=10, alpha=0.1):
    env = NonStationaryBandit(k=k)
    agent = OptimisticGreedyAgent(k=k, initial_value=initial_value, alpha=alpha)
    rewards = np.zeros(steps)

    for t in range(steps):
        action = agent.choose_action()
        reward = env.get_reward(action)
        agent.update(action, reward)
        rewards[t] = reward
        env.step()
    return rewards

def evaluate_optimistic(initial_value, runs=20, steps=200000, k=10, alpha=0.1):
    avg_rewards = []
    for _ in range(runs):
        rewards = run_single_optimistic_experiment(initial_value=initial_value, steps=steps, k=k, alpha=alpha)
        avg_last_100k = np.mean(rewards[-100000:])
        avg_rewards.append(avg_last_100k)
    return np.mean(avg_rewards)

def main():
    plt.figure(figsize=(10, 6))
    
    # epsilon
    epsilons = 2.0 ** np.arange(-7, 0) # [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
    epsilon_results = []
    print("Evaluating Epsilon-Greedy...")
    for eps in epsilons:
        score = evaluate_epsilon(eps, runs=20, steps=200000, alpha=0.1)
        epsilon_results.append(score)
    plt.plot(np.log2(epsilons), epsilon_results, marker='o', label='ε-Greedy (constant step-size α=0.1)')

    # 2. c
    ucb_cs = 2.0 ** np.arange(-4, 3) # [1/16, 1/8, 1/4, 1/2, 1, 2, 4]
    ucb_results = []
    print("Evaluating UCB...")
    for c in ucb_cs:
        score = evaluate_ucb(c, runs=20, steps=200000)
        ucb_results.append(score)
    plt.plot(np.log2(ucb_cs), ucb_results, marker='s', label='UCB')

    # 3. alpha
    gradient_alphas = 2.0 ** np.arange(-5, 3) # [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
    gradient_results = []
    print("Evaluating Gradient Bandit...")
    for alpha in gradient_alphas:
        score = evaluate_gradient(alpha, runs=20, steps=200000)
        gradient_results.append(score)
    plt.plot(np.log2(gradient_alphas), gradient_results, marker='^', label='Gradient Bandit')

    # 4. Q0
    initial_values = 2.0 ** np.arange(-2, 3) # [1/4, 1/2, 1, 2, 4]
    optimistic_results = []
    print("Evaluating Optimistic Greedy...")
    for q0 in initial_values:
        score = evaluate_optimistic(q0, runs=20, steps=200000, alpha=0.1)
        optimistic_results.append(score)
    plt.plot(np.log2(initial_values), optimistic_results, marker='d', label='Optimistic Greedy')
    plt.xlabel("Parameter (log2 scale)")
    plt.ylabel("Average reward\n(last 100,000 steps)")
    plt.title("Comparison of Bandit Algorithms\n(Nonstationary 10-Armed Testbed)")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
