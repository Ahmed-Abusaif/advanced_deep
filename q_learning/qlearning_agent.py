import numpy as np

class QLearningAgent:
    def __init__(self, grid_size, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay=True):
        self.q_table = np.zeros((grid_size, grid_size, 4))  # states x states x actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.grid_size = grid_size
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        return np.argmax(self.q_table[state[0]][state[1]])
    
    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0]][next_state[1]])
        td_target = reward + self.gamma * self.q_table[next_state[0]][next_state[1]][best_next_action]
        td_error = td_target - self.q_table[state[0]][state[1]][action]
        self.q_table[state[0]][state[1]][action] += self.alpha * td_error
    
    def decay_epsilon(self, episode, total_episodes):
        if self.epsilon_decay:
            self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                          np.exp(-5.0 * episode / total_episodes)
