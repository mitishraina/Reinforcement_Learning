import gymnasium as gym
import numpy as np
import os
import pickle as pkl


cliffEnv = gym.make("CliffWalking-v1", render_mode=None)

q_table = np.zeros(shape=(cliffEnv.observation_space.n, cliffEnv.action_space.n))

def policy(obs, epsilon=0.1):
    # action = int(np.argmax(q_table[obs]))
    if np.random.random() < epsilon:
        return cliffEnv.action_space.sample()
    return int(np.argmax(q_table[obs]))

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500

obs = cliffEnv.reset()

for episode in range(NUM_EPISODES):
    obs, _ = cliffEnv.reset()
    action = policy(obs, epsilon=EPSILON)
    
    done = False
    total_reward = 0
    episode_length = 0

    while not done:
        next_obs, reward, terminated, truncated, info = cliffEnv.step(action)
        next_action = policy(next_obs, epsilon=EPSILON)
        
        q_table[obs][action] += ALPHA * (reward + GAMMA * q_table[next_obs][next_action] - q_table[obs][action])
        
        obs = next_obs
        action = next_action
        
        done = terminated or truncated
        
        total_reward += reward
        episode_length += 1
    print(f"Episode: {episode + 1}, Episode Length: {episode_length},Total Reward: {total_reward}")

cliffEnv.close()
os.makedirs("sarsa", exist_ok=True)
with open("sarsa/sarsa_q_table.pkl", "wb") as f:
    pkl.dump(q_table, f)
    print("Q-table saved successfully!")

print("Training complete! bitch")