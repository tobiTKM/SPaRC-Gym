from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import pandas as pd
import yaml

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
        self.max_steps = 2000
        self.current_step = 0
        
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
            
        obs_array : 3D array
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
        
        self._agent_location = np.array([self.start_location[1], self.start_location[0]], dtype=np.int32)
        self._target_location = np.array([self.target_location[1], self.target_location[0]], dtype=np.int32)
        
        # Turn it into a 3D array
        keys = self.obs_array.keys()
        # Guarantee that the first 4 keys are always the same
        feature_names = ['visited', 'gaps', 'agent_location', 'target_location']
        for key in keys:
            if key not in feature_names:
                feature_names.append(key)

        if len(feature_names) != self.unique_properties:
            raise ValueError(f"Number of unique properties does not match the number of features in the observation array. Found {len(feature_names)} features, expected {self.unique_properties}.")
        
        self.obs_array = np.stack([self.obs_array[name] for name in feature_names], axis=0)
        
        # Mark the starting location as visited and set the agent's and target's positions in the observation array
        self.obs_array[0][self._agent_location[0]][self._agent_location[1]] = 1.0
        self.obs_array[2][self._agent_location[0]][self._agent_location[1]] = 1.0
        self.obs_array[3][self._target_location[0]][self._target_location[1]] = 1.0
        
        # Define the observation space for the environment
        # Shape: (number of unique properties, grid width, grid height)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.unique_properties, self.x_size, self.y_size),
            dtype=np.float32,
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
             
    def _get_info(self):
        # Empty for now, but can be used to return extra information about the environment
        return {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Move to the next puzzle
        self.current_puzzle_index = (self.current_puzzle_index + 1) % len(self.puzzles)
        
        self.current_step = 0
        
        # Load the next puzzle
        self._load_puzzle(self.current_puzzle_index)
        
        # Return the initial observation for the new puzzle
        return self._get_obs(), self._get_info()
    
    
    def step(self, action):
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        direction = self._action_to_direction[action]
        
        # np.clip to make sure we don't go out of bounds
        agent_location_temp = np.clip(self._agent_location + direction, [0, 0], [self.y_size - 1, self.x_size - 1])
        
        if self.obs_array[0][agent_location_temp[0], agent_location_temp[1]] == 0.0 and self.obs_array[1][agent_location_temp[0], agent_location_temp[1]] == 0.0:
            # If the next location is not a gap or already visited Cell
            self.obs_array[2][self._agent_location[0]][self._agent_location[1]] = 0.0
            self._agent_location = agent_location_temp
            
            # Update the agent's location in the observation
            self.obs_array[0][self._agent_location[0]][self._agent_location[1]] = 1.0
            self.obs_array[2][self._agent_location[0]][self._agent_location[1]] = 1.0
            
            # Update the path 
            self.path.append(self._agent_location)
            
        else:
            # If the next location is a gap or already visited Cell we do not move
            # and we keep the previous location
            self._agent_location = self._agent_location
            
        
        # An episode is done if the agent has reached the target, does not mean success
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        reward = 0
        # Binary sparse rewards
        if terminated or truncated:
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
        info = self._get_info()
        # If the episode fails for reasons other than reaching the target or failing; Right now not used yet
        return observation, reward, terminated, truncated, info
    

# Idea: What is better: To disallw moves after the action or straight up not giving them the option?
# Idea: Maybe also add an Info fucntion on top with extra information
# need to add pygame(human interpretation)
# Can i always expect the df in the same shape?
