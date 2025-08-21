import gymnasium as gym
import pandas as pd
import gymnasium_env_for_SPaRC
from human_play import play_human
from datasets import load_dataset

''' 
This script initializes the Gym environment and runs a sample episode.
'''

# Load the dataset containing puzzles, here used is the SPaRC dataset.
ds = load_dataset("lkaesberg/SPaRC", 'all', split="test")
df = ds.to_pandas()

# Initialize the Gym environment with the loaded puzzles
env = gym.make("env-SPaRC-v1", puzzles=df, render_mode='human', observation='new', traceback=True, max_steps=1000)

# If you want to play the game using human inputs, use the play_human function. 
# render_mode can now be set to either 'human' or 'llm' and both will work.

obs, reward, info = play_human(env)
env.close()

print(f"Reward: {reward}, info: {info['rule_status']}")

'''
# Otherwise, you can run a sample episode automatically by uncommenting the following lines:

# The reset method resets the puzzle and returns the initial observation and info.
# Has to be called before starting the episode. 
obs, info = env.reset()

while True:
    # Sample a random action from the action space
    # Can be replaced with a specific action if desired.
    action = env.action_space.sample()
    
    # Step the environment with the selected action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Check if the episode has ended
    if terminated or truncated:
        print(f"Reward: {reward}, info: {info}")
        env.close()
        break
'''
