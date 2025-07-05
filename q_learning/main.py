from gridworld import GridWorld
from qlearning_agent import QLearningAgent
from utils import visualize_policy, plot_rewards
import numpy as np  

def train(env, agent, episodes):
    rewards_history = []
    steps_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon(episode, episodes)
        rewards_history.append(total_reward)
        steps_history.append(env.current_step)
        
    return rewards_history, steps_history

def main():
    # Configuration sets
    configs = [
        {"alpha": 0.1, "gamma": 0.9, "epsilon_start": 1.0, "epsilon_end": 0.01},
        {"alpha": 0.5, "gamma": 0.95, "epsilon_start": 1.0, "epsilon_end": 0.1},
        {"alpha": 0.9, "gamma": 0.99, "epsilon_start": 0.5, "epsilon_end": 0.5, "epsilon_decay": False}
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}:")
        print(config)
        
        env = GridWorld()
        agent = QLearningAgent(
            grid_size=4,
            alpha=config["alpha"],
            gamma=config["gamma"],
            epsilon_start=config["epsilon_start"],
            epsilon_end=config["epsilon_end"],
            epsilon_decay=config.get("epsilon_decay", True)
        )
        
        rewards_history, steps_history = train(env, agent, episodes=1000)
        
        print("\nFinal Policy:")
        visualize_policy(agent.q_table)
        plot_rewards(rewards_history, f"Learning Curve - Configuration {i+1}")
        
        print(f"Average steps to goal (last 100 episodes): {np.mean(steps_history[-100:]):.2f}")
        print(f"Average reward (last 100 episodes): {np.mean(rewards_history[-100:]):.2f}")

if __name__ == "__main__":
    main()
