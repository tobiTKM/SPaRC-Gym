from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import pandas as pd
import yaml

splits = {'train': 'puzzle_all_train.jsonl', 'test': 'puzzle_all_test.jsonl'}
df = pd.read_json("hf://datasets/lkaesberg/SPaRC/" + splits["train"], lines=True)

from Dataset_Loader import process_puzzles
puzzles = process_puzzles(df)

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    

class WitnessEnv(gym.Env):
    def __init__(self, puzzles=None):
        # Load the puzzles
        self.puzzles = puzzles if puzzles is not None else ValueError("No puzzles provided")
        self.current_puzzle_index = 0
        
        # Load the first puzzle
        self._load_puzzle(self.current_puzzle_index) 
       
    def _load_puzzle(self, index):
        '''
        Function to load a puzzle from the dataset
        and initialize the environment variables
        aswell as the observation and action spaces
        
        Parameters:
        index : int
            The index of the puzzle to load from the dataset.
            
        --------------
        puzzle variables:
        
        x_size : int
            The x size of the puzzle
            
        y_size : int
            The y size of the puzzle
            
        obs_array : dictionary of 2D one-hot encoded np.arrays
            The observation array of the puzzle
            
        unique_properties : int
            The number of unique properties(star,polyshape, Co.) in the puzzle
        
        start_location : tuple
            The starting location of the agent in the puzzle
            
        target_location : tuple
            The target location of the agent in the puzzle
        
        solution_paths : list
            The solution paths of the puzzle
        
        solution_count : int
            The number of solution paths of the puzzle
        
        '''
        puzzle = self.puzzles[index]
        
        self.x_size = puzzle['x_size']
        self.y_size = puzzle['y_size']
        
        self.obs_array = puzzle['obs_array']
        self.unique_properties = puzzle['unique_properties']
        
        self.start_location = puzzle['start_location']
        self.target_location = puzzle['target_location']
        
        self.solution_paths = puzzle['solution_paths']
        self.solution_count = puzzle['solution_count']
        
        # Initialize the agent's path with the starting location
        self.path = [self.start_location]
        
        self._agent_location = np.array([self.start_location[0], self.start_location[1]], dtype=np.int32)
        self._target_location = np.array([self.target_location[0], self.target_location[1]], dtype=np.int32)
        
        # Mark the starting location as visited and set the agent's and target's positions in the observation array
        self.obs_array['visited'][self._agent_location[0], self._agent_location[1]] = 1
        self.obs_array['agent_location'][self._agent_location[0], self._agent_location[1]] = 1
        self.obs_array['target_location'][self._target_location[0], self._target_location[1]] = 1
        
        # Define the observation space for the environment
        # Shape: (number of unique properties, grid width, grid height)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.unique_properties, self.x_size, self.y_size),
            dtype=np.int32,
        )
        
        # Define the action space (4 discrete actions: right, up, left, down)
        self.action_space = gym.spaces.Discrete(4)
        # Map actions to directions (e.g., right -> [1, 0], up -> [0, 1], etc.)
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }
        
    
    def _get_obs(self):
        # Just returns the current observation array, that always gets updated immediately
        return self.obs_array
             
    
    def reset(self):
        # Move to the next puzzle
        self.current_puzzle_index = (self.current_puzzle_index + 1) % len(self.puzzles)
        
        # Load the next puzzle
        self._load_puzzle(self.current_puzzle_index)
        
        # Return the initial observation for the new puzzle
        return self._get_obs()
    
    
    def step(self, action):
        
        direction = self._action_to_direction[action]
        
        # np.clip to make sure we don't go out of bounds
        agent_location_temp = np.clip(self._agent_location + direction, 0, max(self.x_size, self.y_size) - 1)
        
        if self.obs_array['visited'][agent_location_temp[0], agent_location_temp[1]] == 0 and self.obs_array['gaps'][agent_location_temp[0], agent_location_temp[1]] == 0:
            # If the next location is not a gap or already visited Cell
            self.obs_array['agent_location'][self._agent_location[0], self._agent_location[1]] = 0
            self._agent_location = agent_location_temp
            
            # Update the agent's location in the observation
            self.obs_array['visited'][self._agent_location[0], self._agent_location[1]] = 1
            self.obs_array['agent_location'][self._agent_location[0], self._agent_location[1]] = 1
            
            # Update the path 
            self.path.append(self._agent_location)
            
        else:
            # If the next location is a gap or already visited Cell we do not move
            # and we keep the previous location
            self._agent_location = self._agent_location
            
        
        # An episode is done if the agent has reached the target, does not mean success
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        # Binary sparse rewards
        if terminated:
            for i in range(self.solution_count):
                if np.array_equal(self.path, self.solution_paths[i]):
                    reward = 1
                    break
            if reward != 1:
                reward = 0
        else:
            reward = 0
            
        # Update the observation
        observation = self._get_obs()

        return observation, reward, terminated