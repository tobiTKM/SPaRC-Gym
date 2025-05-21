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
        '''
        Function to initialize the Witness Environment
        and loads the first puzzle from the dataset
        Parameters:
        puzzles : list
        A list of dictionaries containing the puzzles
        '''
        
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
        
        difficulty : int
            The difficulty of the puzzle
            
        x_size : int
            The x size of the puzzle
            
        y_size : int
            The y size of the puzzle
            
        obs_array : dict; Dictionary of 2D arrays
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
        
        self.difficulty = puzzle['difficulty']
        
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

        # Mark the starting location as visited and set the agent's and target's positions in the observation array
        self.obs_array['visited'][self._agent_location[0], self._agent_location[1]] = 1.0
        self.obs_array['agent_location'][self._agent_location[0], self._agent_location[1]] = 1.0
        self.obs_array['target_location'][self._target_location[0], self._target_location[1]] = 1.0
        
        # Define the observation space for the environment
        keys = list(self.obs_array.keys())
        self.observation_space = spaces.Dict({
            key: spaces.Box(low=0, high=1, shape=(self.y_size, self.x_size), dtype=np.int32)
            for key in keys
        }   
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
        '''
        Function to return the current observation of the puzzle
        Returns:
        obs : dict; dictionary of 2D arrays
            A dictionary containing the current observation of the puzzle
        '''
        return self.obs_array
             
    def _get_info(self):
        '''
        Function to return extra information of the current puzzle
        
        Returns:
        info : dict
            A dictionary containing the following information:
            - solution_count: The number of solutions for the current puzzle
            - difficulty: The difficulty of the current puzzle
            - grid_y_size: The y size of the current puzzle
            - grid_x_size: The x size of the current puzzle
            - legal_actions: The legal actions for the current state of the agent
            - current_step: The current step of the agent
        '''
        info = {"solution_count": self.solution_count,
        "difficulty": self.difficulty,
        "grid_y_size": self.y_size,
        "grid_x_size": self.x_size,
        "legal_actions": self.get_legal_actions(),
        "current_step": self.current_step
        }
        return info
    
    def get_legal_actions(self):
        '''
        Function to get the legal actions for the current state of the agent
        
        Returns:
        legal : list
            A list of legal actions for the current state of the agent
        '''
        legal = []
        
        for action, direction in self._action_to_direction.items():
            next_loc = self._agent_location + direction
            # np.clip to make sure we don't go out of bounds
            agent_location_temp = np.clip(next_loc, [0, 0], [self.y_size - 1, self.x_size - 1])
            # Check if the next location is not a gap or already visited Cell
            if self.obs_array['visited'][agent_location_temp[0], agent_location_temp[1]] == 0 and self.obs_array['gaps'][agent_location_temp[0], agent_location_temp[1]] == 0:
                legal.append(action)
            
        return legal
    
    def reset(self, seed=None, options=None):
        '''
        Function to reset the environment and load the next puzzle
        Parameters:
        seed : int
            The seed for the random number generator
        options : dict
            Additional options for resetting the environment
            Not used yet
        ----------
        Returns:
        obs : dict; dictionary of 2D arrays
            A dictionary containing the current observation of the puzzle
        info : dict
            A dictionary containing the extra information of the current puzzle
        '''
        super().reset(seed=seed)
        
        # Move to the next puzzle
        self.current_puzzle_index = (self.current_puzzle_index + 1) % len(self.puzzles)
        
        self.current_step = 0
        
        # Load the next puzzle
        self._load_puzzle(self.current_puzzle_index)
        
        # Return the initial observation for the new puzzle
        return self._get_obs(), self._get_info()
    
    
    def step(self, action):
        '''
        Function to take a step in the environment
        Parameters:
        action : int
            The action to take in the environment
        ----------
        Returns:
        obs : dict; dictionary of 2D arrays
            A dictionary containing the current observation of the puzzle
        reward : int
            The reward for taking the action
        terminated : bool
            Whether the episode has terminated
        truncated : bool
            Whether the episode has been truncated
        info : dict
            A dictionary containing the extra information of the current puzzle
        '''
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # If there are no legal actions left, the episode is truncated
        if self.get_legal_actions() == []:
            truncated = True
        
        direction = self._action_to_direction[action]
        
        # np.clip to make sure we don't go out of bounds
        agent_location_temp = np.clip(self._agent_location + direction, [0, 0], [self.y_size - 1, self.x_size - 1])
        
        if self.obs_array['visited'][agent_location_temp[0], agent_location_temp[1]] == 0 and self.obs_array['gaps'][agent_location_temp[0], agent_location_temp[1]] == 0:
            # If the next location is not a gap or already visited Cell
            self.obs_array['agent_location'][self._agent_location[0]][self._agent_location[1]] = 0
            self._agent_location = agent_location_temp
            
            # Update the agent's location in the observation
            self.obs_array['visited'][self._agent_location[0]][self._agent_location[1]] = 1
            self.obs_array['agent_location'][self._agent_location[0]][self._agent_location[1]] = 1
            
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
    

# Idea: What is better: To disallow moves after the action or straight up not giving them the option? --> Straight up not giving them the option
# Not possible to straight up not give them the option, because the Action Space needs to be static, and can not be adjusted for each step
# Made a function that returns the legal actions for the current state of the agent that is returned in the info dict
# Also if no legal actions are left, the episode is truncated 

# Big Problem with the current Observation Space:
# The agent can not distinguish/learn (except for the first 4; because they are always the same(visited,gaps,agent_location,target_location)) the different channels
# in the observation array 
# If we made the channels for every puzzle the same, that would kinda fix it --> but then we would have a lot of empty arrays
# dictionary of 2d arrays could fix that
# Another Idea: Dictionary but not of 2D arrays, but of the points aka dot: (7,5),(3,2) as an example

# need to add pygame(human interpretation) 
# Docstrings; clear Documentation
# Do Readme file

# klares feedback (return)
