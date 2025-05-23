from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import pandas as pd
import yaml
import gymnasium_env
import warnings

from Dataset_Loader import process_puzzles
from human_play import play_human

warnings.filterwarnings("ignore", category=UserWarning)

''' 
This script initializes the Witness environment, processes puzzle data, and runs a sample episode.
'''

splits = {'train': 'puzzle_all_train.jsonl', 'test': 'puzzle_all_test.jsonl'}
df = pd.read_json("hf://datasets/lkaesberg/SPaRC/" + splits["train"], lines=True)

puzzles = process_puzzles(df)


env = gym.make("Witness-v0", puzzles=puzzles)

# 9, 52
obs, reward, info = play_human(env)
    
print(f"Reward: {reward}, info: {info}")