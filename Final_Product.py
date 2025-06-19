import gymnasium as gym
import pandas as pd
import gymnasium_env_for_SPaRC
from human_play import play_human

''' 
This script initializes the Gym environment and runs a sample episode.
'''

# Load the dataset containing puzzles, here used is the SPaRC dataset.
splits = {'train': 'puzzle_all_train.jsonl', 'test': 'puzzle_all_test.jsonl'}
df = pd.read_json("hf://datasets/lkaesberg/SPaRC/" + splits["train"], lines=True)

# Initialize the Gym environment with the loaded puzzles
env = gym.make("env-SPaRC-v0", puzzles=df, render_mode='human', traceback=True, max_steps=1000)

# If you want to play the game using human inputs, use the play_human function. 
# render_mode needs to be set to 'human' for this to work.
obs, reward, info = play_human(env)
    
print(f"Reward: {reward}, info: {info}")

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
        break
'''
