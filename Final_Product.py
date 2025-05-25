import gymnasium as gym
import pandas as pd
import gymnasium_env
import warnings
import time
from human_play import play_human

warnings.filterwarnings("ignore", category=UserWarning)

''' 
This script initializes the Witness environment and runs a sample episode.
'''

# Load the dataset containing puzzles, here used is the SPaRC dataset.
splits = {'train': 'puzzle_all_train.jsonl', 'test': 'puzzle_all_test.jsonl'}
df = pd.read_json("hf://datasets/lkaesberg/SPaRC/" + splits["train"], lines=True)

# Initialize the Witness environment with the loaded puzzles
env = gym.make("Witness-v0", puzzles=df)

# If you want to play the game using human inputs, use the play_human function.
obs, reward, info = play_human(env)
    
print(f"Reward: {reward}, info: {info}")
print('\n')
print(f"Observation:", obs)

'''
# Otherwise, you can run a sample episode automatically by uncommenting the following lines:
# and replace action with your desired actions.

# The reset method resets the puzzle and returns the initial observation and info.
# Has to be called before starting the episode. 
obs, info = env.reset()

# Visualize the initial state of the environment
# env.render() will display the current state of the puzzle.
env.render()

while True:
    # Sample a random action from the action space
    # Can be replaced with a specific action if desired.
    action = env.action_space.sample()
    
    # Step the environment with the selected action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(1)
    
    # Check if the episode has ended
    if terminated or truncated:
        print(f"Reward: {reward}, info: {info}")
        break
'''