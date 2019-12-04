import matplotlib.pyplot as plt
import numpy as np 
import random
import yaml

class GreedyAgent:
    def __init__(self, k=10, epsilon=0.01):
        """Initialize variables for the greedy agent.

        Keyword arguments:
        k -- the number of bandits (default 10)
        epsilon -- the probability that the agent will select an action randomly (default 0.01)
        """
        self.k = k
        self.epsilon = epsilon
        self.actions = [index for index in range(k)]
        self.true_values = [np.random.normal(0,1) for x in range(k)]
        self.q_values = np.zeros(self.k)
        self.action_count = np.zeros(k).tolist()
        self.rewards = np.zeros(k).tolist()
        self.ave_rewards = []

    def run_agent(self, steps=1000):
        """Run and evaluate the epsilon greedy agent.

        Keyword arguments:
        steps -- the number of steps the agent will take in the learning process (default 1000)
        """
        for step in range(1,steps+1):
            action = get_action(self.epsilon, self.q_values, self.actions)
            get_reward(self.rewards, self.true_values[action], action)
            self.action_count[action] += step
            update_qval(self.q_values,self.action_count, self.rewards, action)
            _qval = [x/y if y else 0 for x,y in zip(self.q_values,self.action_count)]
            self.ave_rewards.append(np.mean(_qval))
            

        # Plot the results    
        x = [x for x in range(steps)]
        y = self.ave_rewards
        plt.figure(figsize=(10, 20))
        plt.plot(x, y)
        plt.xlabel('steps')
        plt.ylabel('average reward')
        plt.title("Epsilon Greedy Multi-Armed Bandit | Number of Bandits: {} Epsilon: {}".format(self.k, self.epsilon))
        plt.show() 

def get_action(epsilon, q_values, actions):
    """Return the action to be taken by the agent.

    Keyword arguments:
    epsilon -- the probability that the agent will select an action randomly.
    q_values -- the estimated values of the actions.
    actions -- the possible moves/actions that the agent can take in the environment.
    """
    val = np.random.randn()
    if val < epsilon:
        action = random.choice(actions)
    else:
        action = np.argmax(q_values)
    return action

def get_reward(rewards, true_value, action):
    """Return the reward of the user for taking a certain action.

    Keyword arguments:
    rewards -- the current rewards prior to taking the action.
    true_value -- the value of the action that the agent is trying to estimate.
    action -- the action taken by the agent that will be evaluated.
    """
    rewards[action] = np.random.normal(true_value,1) 
    return rewards

def update_qval(q_values,action_count, reward, action):
    """Return the updated value estimates of the actions taken.

    Keyword arguments:
    q_values -- the current value estimates of the actions.
    action_count -- the number of times the action has been selected.
    reward -- the reward given after taking an action.
    action -- the current selected action.
    """
    q_values[action] = q_values[action] + ((reward[action]-q_values[action])/action_count[action])
    return q_values

def config_args(config):
    """Return hyperparameters in the configuration file.

    Keyword arguments:
    config -- the configuration file that contains the hyperparameters needed for the program.
    """
    with open('config.yaml', 'rb') as file:
        conf = yaml.safe_load(file)
    return conf['evaluate']

if __name__ == '__main__':
    args = config_args('config.yaml')
    agent = GreedyAgent(args['k'], args['epsilon'])
    agent.run_agent(args['steps'])