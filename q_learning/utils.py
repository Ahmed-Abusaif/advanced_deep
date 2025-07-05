import numpy as np
import matplotlib.pyplot as plt

def visualize_policy(q_table):
    grid_size = q_table.shape[0]
    policy_viz = [['' for _ in range(grid_size)] for _ in range(grid_size)]
    
    arrows = ['↑', '→', '↓', '←']
    
    for i in range(grid_size):
        for j in range(grid_size):
            if i == grid_size-1 and j == grid_size-1:
                policy_viz[i][j] = 'G'
            else:
                action = np.argmax(q_table[i][j])
                policy_viz[i][j] = arrows[action]
    
    for row in policy_viz:
        print(' '.join(row))

def plot_rewards(rewards_history, title):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
