from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import yaml
import math
from .render import HumanRenderer, LLMRenderer

class Actions(Enum):
    """
    Enum class representing the possible actions the agent can take in the environment.

    Actions:
        right (int): Move the agent one step to the right.
        up (int): Move the agent one step upward.
        left (int): Move the agent one step to the left.
        down (int): Move the agent one step downward.
    """
    right = 0
    up = 1
    left = 2
    down = 3
    

class GymEnvSPaRC(gym.Env):
    metadata = {"render_modes": ["human", "llm"], "render_fps": 30}
    def __init__(self, puzzles=None, render_mode=None, observation='new', traceback=False, max_steps=2000):
        '''
        Function to initialize the Witness Environment, processes the puzzles dataset,
        and loads the first puzzle from the dataset
        Parameters:
        puzzles : df
        A pandas DataFrame containing the puzzles to be used in the environment.
        '''
        self.render_mode = render_mode
        self.observation = observation
        self.traceback = traceback
        self.max_steps = max_steps

        # Initialize renderers
        self.human_renderer = None
        self.llm_renderer = None
        if render_mode == "human":
            self.human_renderer = HumanRenderer(scale_factor=3.0)
        elif render_mode == "llm":
            self.llm_renderer = LLMRenderer()

        # Load the puzzles
        self.puzzles = puzzles if puzzles is not None else ValueError("No puzzles provided")
        self.current_puzzle_index = 0
        self.current_step = 0
        
        # Process the puzzles to extract relevant information
        self.puzzles = self.process_puzzles(self.puzzles)
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
        
        polyshapes : dict of 2d arrays
            The polyshapes of the puzzle
        
        x_size : int
            The x size of the puzzle
            
        y_size : int
            The y size of the puzzle
            
        obs_array : dict; Dictionary of 2D arrays
            The observation array of the puzzle
        or if observation == 'SPaRC':
            The observation array in SPaRC format
        
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
        self.polyshapes = puzzle['polyshapes']
        
        self.x_size = puzzle['x_size']
        self.y_size = puzzle['y_size']
        
        self.obs_array = puzzle['obs_array']
        self.color_array = puzzle['color_array']
        self.additional_info = puzzle['additional_info']
        
        if self.observation == 'SPaRC':
            self.observ = puzzle['observ']
                
        self.start_location = puzzle['start_location']
        self.target_location = puzzle['target_location']
        
        self.solution_paths = puzzle['solution_paths']
        self.solution_count = puzzle['solution_count']
        
        # Initialize the agent's path with the starting location
        self.path = [[self.start_location[0], self.start_location[1]]]
        self.normal_reward = 0
        self.outcome_reward = 0
        
        self._agent_location = np.array([self.start_location[1], self.start_location[0]], dtype=np.int32)
        self._target_location = np.array([self.target_location[1], self.target_location[0]], dtype=np.int32)

        # Mark the starting location as visited and set the agent's and target's positions in the observation array
        self.obs_array['visited'][self._agent_location[0], self._agent_location[1]] = 1
        self.obs_array['agent_location'][self._agent_location[0], self._agent_location[1]] = 1
        self.obs_array['target_location'][self._target_location[0], self._target_location[1]] = 1
        
        # Define the observation space for the environment
        if self.observation == 'new':
            keys = list(self.obs_array.keys())
            self.observation_space = spaces.Dict({
                'base': spaces.Dict({key: spaces.Box(low=0, high=1, shape=(self.y_size, self.x_size), dtype=np.int32) for key in keys}),
                'color': spaces.Box(low=0, high=8, shape=(self.y_size, self.x_size), dtype=np.int32),
                'additional_info': spaces.Box(low=0, high=143632, shape=(self.y_size, self.x_size), dtype=np.int64)
            })
        
        elif self.observation == 'SPaRC':
            max_length = sum(len(tok) for row in self.observ for tok in row) \
            + (self.observ.shape[1]-1)*self.observ.shape[0] + (self.observ.shape[0]-1)
            self.observation_space = spaces.Text(max_length=max_length)

        else:
            raise ValueError("Invalid observation type. Choose 'new' or 'SPaRC'.")

        # Define the action space (4 discrete actions: right, up, left, down)
        self.action_space = gym.spaces.Discrete(4)
        # Map actions to directions 
        self._action_to_direction = {
            Actions.right.value: np.array([0, 1]),
            Actions.up.value: np.array([-1, 0]),
            Actions.left.value: np.array([0, -1]),
            Actions.down.value: np.array([1, 0]),
        }
    
    def process_puzzles(self, df):
        """
        Processes a DataFrame of puzzles and returns a list of puzzle dictionaries.

        Parameters:
            df (pd.DataFrame): The DataFrame containing puzzle data.
            
        --------
        
        Returns:
            list: A list of dictionaries, each representing a processed puzzle.
        """
        
        if df is None:
                raise ValueError("No dataframe provided")
            
        puzzles = []

        for i in range(len(df)):

            puzzle = {}
            
            # Extract difficulty
            difficulty = df['difficulty_level'][i]
            puzzle.update({'difficulty': difficulty})
            
            # Extract grid size
            grid_size = df['grid_size'][i]
            x_size = grid_size['width']
            y_size = grid_size['height']
            x_size = x_size + x_size + 1
            y_size = y_size + y_size + 1
            puzzle.update({'x_size': x_size, 'y_size': y_size})
            
            # Extract solution paths
            solution_count = df['solution_count'][i]
            solutions = df['solutions'][i]
            solution_paths = []
            for item in solutions:
                path = [[point["x"], point["y"]] for point in item["path"]]
                solution_paths.append(path)
            puzzle.update({'solution_count': solution_count, 'solution_paths': solution_paths})
            
            # Extract the polyshapes (eg. an L shape)
            polyshapes = df['polyshapes'][i]
            polyshapes_yaml = yaml.safe_load(polyshapes)
            puzzle.update({'polyshapes': polyshapes_yaml})
            
            # Extract start and target locations
            text_visualization = df['text_visualization'][i]
            text_yaml = yaml.safe_load(text_visualization)
            start_location = (text_yaml["puzzle"]["start"]["x"], text_yaml["puzzle"]["start"]["y"])
            target_location = (text_yaml["puzzle"]["end"]["x"], text_yaml["puzzle"]["end"]["y"])
            puzzle.update({'start_location': start_location, 'target_location': target_location})
            
            # Initialize observation arrays
            obs_array = {
                'visited': np.zeros((y_size, x_size), dtype=np.int32),
                'gaps': np.zeros((y_size, x_size), dtype=np.int32),
                'agent_location': np.zeros((y_size, x_size), dtype=np.int32),
                'target_location': np.zeros((y_size, x_size), dtype=np.int32)
            }
            
            color_array = np.zeros((y_size, x_size), dtype=np.int32)
            additional_info = np.zeros((y_size, x_size), dtype=np.int64)
            
            # Extract symbols, colors and additional info 
            for cell in text_yaml["puzzle"]["cells"]:
                properties = cell.get("properties", {})
                count = None
                shape = None
                color = None
                for key, value in properties.items():
                    if key == 'type':
                        if value == 'star' or value == 'square':
                            symbol = f"{value}"
                            color = properties.get('color', '')
                        elif value == 'triangle':
                            symbol = f"{value}"
                            color = properties.get('color', '')
                            count = properties.get('count', '')
                        else:
                            symbol = f"{value}"
                            color = properties.get('color', '')
                            shape = properties.get('polyshape', '')
                        
                    elif key == 'dot':
                        symbol = 'dot'
                    # Add new property to obs_array if not already present
                    if symbol not in obs_array:
                        obs_array.update({symbol: np.zeros((y_size, x_size), dtype=np.int32)})
                        
                    # Update the colors
                    if color:
                        color_to_number = {"red": 1, "blue": 2, "yellow": 3, "green": 4, "black": 5, "purple": 6, "orange": 7, "white": 8} 
                        position = cell.get("position", {}) 
                        x, y = position.get("x"), position.get("y")                  
                        for color_ in color_to_number:
                            if color_ == color:
                                color_array[y][x] = color_to_number[color_]
                            
                    # update additional information
                    if count:
                        position = cell.get("position", {}) 
                        x, y = position.get("x"), position.get("y")  
                        additional_info[y][x] = count
                    elif shape:
                        position = cell.get("position", {}) 
                        x, y = position.get("x"), position.get("y")  
                        additional_info[y][x] = shape
                    
            
            # Populate observation arrays
            for cell in text_yaml["puzzle"]["cells"]:
                position = cell.get("position", {})
                properties = cell.get("properties", {})
                x, y = position.get("x"), position.get("y")

                for key, value in properties.items():
                    if key == 'type':
                        symbol = f"{value}"
                    elif key == 'dot':
                        symbol = 'dot'
                    elif key == 'gap':
                        symbol = 'gaps'
                    # Update the corresponding observation array
                    if symbol in obs_array:
                        obs_array[symbol][y, x] = 1

            x_size = x_size - 1
            y_size = y_size - 1
            # Mark all the green cells as gaps
            for i in range(x_size):
                for j in range(y_size):
                    if i % 2 == 1 and j % 2 == 1:
                        obs_array['gaps'][j, i] = 1
            
            puzzle.update({'obs_array': obs_array})
            puzzle.update({'color_array': color_array})
            puzzle.update({'additional_info': additional_info})

            # If using the SPaRC observation format
            if self.observation == 'SPaRC':
                observ = df['puzzle_array'][i]
                puzzle.update({'observ': observ})

            # Add the processed puzzle to the list
            puzzles.append(puzzle)
        
        return puzzles
    
    def _grid_to_text(grid):
        return "\n".join(" ".join(str(cell) for cell in row) for row in grid)
    
    def _get_obs(self):
        '''
        Function to return the current observation of the puzzle
        if observation == 'new':
        
        Returns: dictionary of:
        dictionary of 2D arrays,
        2D array,
        2D array
        ----------
        {
        obs : dict; dictionary of 2D arrays
            A dictionary containing the current locations of the properties of the puzzle
        color_array : 2D array
            A 2D array containing the colors of the properties in the puzzle
        additional_info : 2D array
            A 2D array containing additional information about the properties in the puzzle
        }

        if observation == 'SPaRC':
        Returns a string representation of the puzzle in the SPaRC format
        '''
        if self.observation == 'new':
            return {'base': self.obs_array, 'color': self.color_array, 'additional_info': self.additional_info}
        
        elif self.observation == 'SPaRC':
            observ = self._grid_to_text(self.obs_array)
            return {'observ': observ}
        
        else:
            raise ValueError("Invalid observation type. Choose 'new' or 'SPaRC'.")

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
        "current_step": self.current_step,
        "agent_location": self._agent_location,
        "Rewards": {"normal_reward": self.normal_reward, "outcome_reward": self.outcome_reward}
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
            # Check if the next location is not a gap
            if self.obs_array['gaps'][agent_location_temp[0], agent_location_temp[1]] == 0:
                if self.obs_array['visited'][agent_location_temp[0], agent_location_temp[1]] == 1:
                    if self.traceback:
                        if len(self.path) >= 2:
                            last_loc = np.array([self.path[-2][1], self.path[-2][0]], dtype=np.int32)
                            if np.array_equal(last_loc, agent_location_temp):
                                if np.array_equal(next_loc, agent_location_temp): 
                                    legal.append(action)
                else:
                    if np.array_equal(next_loc, agent_location_temp):
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
        
        # Also possible to randomly select a puzzle
        # self.current_puzzle_index = np.random.randint(0, len(self.puzzles))
        
        self.current_step = 0
        
        # Load the next puzzle
        self._load_puzzle(self.current_puzzle_index)
        
        # Visualize the initial state of the environment
        if self.render_mode == "human":
            if self.human_renderer is None:
                self.human_renderer = HumanRenderer(scale_factor=3.0)
            self.render()
        elif self.render_mode == "llm":
            if self.llm_renderer is None:
                self.llm_renderer = LLMRenderer()
            self.render()
        
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
        
        orig_loc = self._agent_location.copy()
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # If the action is not in the legal actions, we do not move
        if action in self.get_legal_actions():
            direction = self._action_to_direction[action]
            agent_location_temp = self._agent_location + direction
            
            if self.obs_array['visited'][agent_location_temp[0], agent_location_temp[1]] == 1:
                if self.traceback:
                    last_loc = np.array([self.path[-2][1], self.path[-2][0]], dtype=np.int32)
                    if np.array_equal(last_loc, agent_location_temp): 
                        # If the next location is already visited and is the last location, we are allowed to move back
                        self.obs_array['agent_location'][self._agent_location[0]][self._agent_location[1]] = 0
                        self.obs_array['visited'][self._agent_location[0]][self._agent_location[1]] = 0
                        self._agent_location = agent_location_temp

                        # Update the SPaRC observation if it is active
                        if self.observation == 'SPaRC':
                            if self.obs_array['gaps'][self._agent_location[0]][self._agent_location[1]] == 1:
                                self.observ = [self._agent_location[1], self._agent_location[0]] = '.'
                            else:
                                self.observ = [self._agent_location[1], self._agent_location[0]] = '+'
                        
                        # Update the agent's location in the observation
                        self.obs_array['visited'][self._agent_location[0]][self._agent_location[1]] = 1
                        self.obs_array['agent_location'][self._agent_location[0]][self._agent_location[1]] = 1

                        # Also update the SPaRC observation if it is active
                        if self.observation == 'SPaRC':
                            self.observ[self._agent_location[1]][self._agent_location[0]] = 'L'

                        # Update the path
                        del self.path[-1]
            else:
                self.obs_array['agent_location'][self._agent_location[0]][self._agent_location[1]] = 0

                # Update the SPaRC observation if it is active
                if self.observation == 'SPaRC':
                    self.observ[self._agent_location[1]][self._agent_location[0]] = 'V'

                self._agent_location = agent_location_temp
                
                # Update the agent's location in the observation
                self.obs_array['visited'][self._agent_location[0]][self._agent_location[1]] = 1
                self.obs_array['agent_location'][self._agent_location[0]][self._agent_location[1]] = 1

                # Also update the SPaRC observation if it is active
                if self.observation == 'SPaRC':
                    self.observ[self._agent_location[1]][self._agent_location[0]] = 'L'
                
                # Update the path
                path = [self._agent_location[1], self._agent_location[0]]
                self.path.append(path)
          
        
        # An episode is done if the agent has reached the target, does not mean success
        terminated = np.array_equal(self._agent_location, self._target_location)
        
        # If there are no legal actions left (for the next step), the episode is truncated
        if self.get_legal_actions() == []:
            truncated = True
        
        # Reward logic:
        # Have an outcome reward and a normal reward
        # The normal reward is updated during the episode, the outcome reward is only updated at the end of the episode
        if terminated or truncated:
            for i in range(self.solution_count):
                if np.array_equal(self.path, self.solution_paths[i]):
                    self.outcome_reward = 1
                    self.normal_reward = 1
                    break
                
            if self.outcome_reward != 1:
                self.outcome_reward = -1
                self.normal_reward = -1
        else:
            self.outcome_reward = 0
            if not np.array_equal(orig_loc, self._agent_location):
                for i in range(self.solution_count):
                    current_solution_path = self.solution_paths[i]
                    if self.is_on_solution_path(self.path, current_solution_path):
                        self.normal_reward += 0.01
                        break

        
        # Update the observation
        observation = self._get_obs()
        info = self._get_info()

        # Unfortunately, I have to return the normal reward here, since gymnasium expects a reward to be a scalar value, not a dictionary of scalar values
        reward = self.normal_reward
        
        # Visualize the current state of the environment
        if self.render_mode == "human" or self.render_mode == "llm":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def is_on_solution_path(self, current_path, solution_path):
        """
        Checks if the current path is still on the solution path.

        Args:
            current_path (list): The path taken so far.
            solution_path (list): The correct solution path.

        Returns:
            bool: True if the current path is on the solution path, False otherwise.
        """
        # If the current path is longer than the solution path, return False
        if len(current_path) > len(solution_path):
            return False

        # Compare each step in the current path with the solution path
        for i in range(len(current_path)):
            if current_path[i] != solution_path[i]:
                return False

        return True

    def render(self):
        """
        Renders the environment using the appropriate renderer based on render_mode.
        
        Returns:
            str or None: For LLM mode, returns a text representation. For human mode, returns None.
        """
        if self.render_mode == "human" and self.human_renderer is not None:
            return self.human_renderer.render(
                self.obs_array, 
                self.color_array, 
                self.additional_info, 
                self.polyshapes, 
                self.x_size, 
                self.y_size,
                self.start_location,
                self.target_location,
                self.path
            )
        elif self.render_mode == "llm" and self.llm_renderer is not None:
            return self.llm_renderer.render(
                self.obs_array, 
                self.color_array, 
                self.additional_info, 
                self.polyshapes, 
                self.x_size, 
                self.y_size
            )
        else:
                         # No rendering or unsupported mode
             return None
    
    def close(self):
        """
        Close the environment and cleanup any resources.
        """
        if self.human_renderer is not None:
            self.human_renderer.close()
            self.human_renderer = None
            
        if self.llm_renderer is not None:
            self.llm_renderer.close()
            self.llm_renderer = None

                
                
