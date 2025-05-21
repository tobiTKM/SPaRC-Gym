from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import pandas as pd
import yaml
import gymnasium_env
import warnings

''' 
This script initializes the Witness environment, processes puzzle data, and runs a sample episode.
'''

splits = {'train': 'puzzle_all_train.jsonl', 'test': 'puzzle_all_test.jsonl'}
df = pd.read_json("hf://datasets/lkaesberg/SPaRC/" + splits["train"], lines=True)

from Dataset_Loader import process_puzzles
puzzles = process_puzzles(df)

warnings.filterwarnings("ignore", category=UserWarning)

env = gym.make("Witness-v0", puzzles=puzzles)

obs, info = env.reset()  
print(f"Initial Observation: {obs}")

done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
print(f"Reward: {reward}, info: {info} Done: {done}")
