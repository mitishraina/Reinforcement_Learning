import gymnasium as gym
import pickle as pkl
import numpy as np
import cv2
from world import initalize_world, put_agent


with open("q_learning/q_learning_q_table.pkl", "rb") as f:
    q_table = pkl.load(f)
 
 
cliffEnv = gym.make("CliffWalking-v1", render_mode="human")
 
def policy(obs):
    return int(np.argmax(q_table[obs]))
 
for episode in range(5): 
    done = False
    frame = initalize_world()
    obs, _ = cliffEnv.reset()
    
    total_reward = 0
    episode_length = 0

    while not done:
        frame2 = put_agent(frame.copy(), obs)
        cv2.imshow("cliff world", frame2)
        cv2.waitKey(250)
        action = policy(obs)
        obs, reward, terminated, truncated, info = cliffEnv.step(action)
        done = terminated or truncated
        
        episode_length += 1
        total_reward += reward
        
    print(f"Episode: {episode + 1}, Episode Length: {episode_length},Total Reward: {total_reward}")

cliffEnv.close()
cv2.destroyAllWindows()
    