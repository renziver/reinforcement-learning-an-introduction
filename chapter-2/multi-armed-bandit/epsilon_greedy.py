import matplotlib.pyplot as plt
import numpy as np 
import random
import yaml

class GreedyAgent:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.actions = [index for index in range(k)]
        self.true_values = [np.random.normal(0,1) for x in range(k)]
        self.q_values = np.zeros(self.k)
        self.nth_action = np.zeros(k).tolist()
        self.rewards = np.zeros(k).tolist()
        self.ave_rewards = []

    def run(self, steps):
        for step in range(1,steps+1):
            selected_act = select_action(self.epsilon, self.q_values, self.actions)
            reward(self.rewards, self.true_values[selected_act], selected_act)
            increase_nth(self.nth_action, selected_act)
            update_qval(self.q_values,self.nth_action, self.rewards, selected_act)
            _qval = [x/y if y else 0 for x,y in zip(self.q_values,self.nth_action)]
            self.ave_rewards.append(np.mean(_qval))
            
        x = [x for x in range(steps)]
        y = self.ave_rewards
        plt.figure(figsize=(10, 20))
        plt.plot(x, y)
        plt.xlabel('steps')
        plt.ylabel('average reward')
        plt.title("Epsilon Greedy Multi-Armed Bandit | Number of Bandits: {} Epsilon: {}".format(self.k, self.epsilon))
        plt.show() 

def select_action(epsilon, q_values, actions):
    val = np.random.randn()
    if val < epsilon:
        action = random.choice(actions)
    else:
        action = np.argmax(q_values)
    return action

def reward(rewards,true_values, action):
    rewards[action] = np.random.normal(true_values,1) 
    return rewards

def increase_nth(nth_action, action):
    nth_action[action] += 1
    return nth_action

def update_qval(q_values,nth_action, reward, action):
    q_values[action] = q_values[action] + ((reward[action]-q_values[action])/nth_action[action])
    return q_values

def eval_arguments(config):
    with open('config.yaml', 'rb') as f:
        conf = yaml.safe_load(f)
    return conf['evaluate']

if __name__ == '__main__':
    args = eval_arguments('config.yaml')
    agent = GreedyAgent(args['k'], args['epsilon'])
    agent.run(args['steps'])