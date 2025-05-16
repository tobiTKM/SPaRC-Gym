from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import pandas as pd
import yaml
import gymnasium_env
import warnings

splits = {'train': 'puzzle_all_train.jsonl', 'test': 'puzzle_all_test.jsonl'}
df = pd.read_json("hf://datasets/lkaesberg/SPaRC/" + splits["train"], lines=True)

from Dataset_Loader import process_puzzles
puzzles = process_puzzles(df)

warnings.filterwarnings("ignore", category=UserWarning)

env = gym.make("Witness-v0", puzzles=puzzles)