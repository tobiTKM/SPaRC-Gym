from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import pandas as pd
import yaml
import math

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    

class WitnessEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
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
        
        polyshapes : dict of 2d arrays
            The polyshapes of the puzzle
        
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
        self.polyshapes = puzzle['polyshapes']
        
        self.x_size = puzzle['x_size']
        self.y_size = puzzle['y_size']
        
        self.obs_array = puzzle['obs_array']
        self.unique_properties = puzzle['unique_properties']
        
        self.start_location = puzzle['start_location']
        self.target_location = puzzle['target_location']
        
        self.solution_paths = puzzle['solution_paths']
        self.solution_count = puzzle['solution_count']
        
        # Initialize the agent's path with the starting location
        self.path = [[self.start_location[0], self.start_location[1]]]
        
        self._agent_location = np.array([self.start_location[1], self.start_location[0]], dtype=np.int32)
        self._target_location = np.array([self.target_location[1], self.target_location[0]], dtype=np.int32)

        # Mark the starting location as visited and set the agent's and target's positions in the observation array
        self.obs_array['visited'][self._agent_location[0], self._agent_location[1]] = 1
        self.obs_array['agent_location'][self._agent_location[0], self._agent_location[1]] = 1
        self.obs_array['target_location'][self._target_location[0], self._target_location[1]] = 1
        
        # Define the observation space for the environment
        keys = list(self.obs_array.keys())
        self.observation_space = spaces.Dict({
            key: spaces.Box(low=0, high=1, shape=(self.y_size, self.x_size), dtype=np.int32)
            for key in keys
        }   
        )
        
        # Define the action space (4 discrete actions: right, up, left, down)
        self.action_space = gym.spaces.Discrete(4)
        # Map actions to directions 
        self._action_to_direction = {
            Actions.right.value: np.array([0, 1]),
            Actions.up.value: np.array([-1, 0]),
            Actions.left.value: np.array([0, -1]),
            Actions.down.value: np.array([1, 0]),
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
            path = [self._agent_location[1], self._agent_location[0]]
            self.path.append(path)
            
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
    

    def render(self):
        """
        Visualizes the current state of the environment using pygame.
        """
        cell_size = 40
        margin = 2
        width = self.x_size * cell_size
        height = self.y_size * cell_size

        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("WitnessEnv Visualization")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))  # White background

        # Draw grid and elements
        for y in range(self.y_size):
            for x in range(self.x_size):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size - margin, cell_size - margin)
                color = (200, 200, 200)  # Default: light gray

                # Draw visited cells
                if self.obs_array["visited"][y, x]:
                    color = (180, 255, 180) # Light green for visited cells, since they can not be visited again but are not gaps
                # Draw gaps
                if self.obs_array["gaps"][y, x]:
                    color = (0, 128, 0) # Green for gaps
                # Draw agent
                if self.obs_array["agent_location"][y, x]:
                    color = (0, 0, 255) # Blue for agent
                # Draw target
                if self.obs_array["target_location"][y, x]:
                    color = (255, 0, 0) # Red for target

                pygame.draw.rect(self.screen, color, rect)

         # Draw other properties
        for prop, array in self.obs_array.items():
            if prop in ["visited", "gaps", "agent_location", "target_location"]:
                continue  # Skip already visualized properties

            for y in range(self.y_size):
                for x in range(self.x_size):
                    if array[y, x]:  # If the property exists at this cell
                        center = (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)

                        parts = prop.split("_")
                        prop_type = parts[0]  # e.g., "star", "poly", "triangle", "dot"
                        color = self._get_color_from_name(parts)  # Extract color

                        
                        if prop_type == "star":
                            self._draw_star(self.screen, color, center, cell_size // 4)
                            
                        elif prop_type == "poly":
                            shape = parts[1]
                            shape_array = self.polyshapes[shape]
                            top_left = (x * cell_size, y * cell_size)
                            self._draw_polyshape(self.screen, shape_array, top_left, cell_size, color)
                        
                        elif prop_type == "ylop":
                            shape = parts[1]
                            shape_array = self.polyshapes[shape]
                            top_left = (x * cell_size, y * cell_size)
                            self._draw_polyshape(self.screen, shape_array, top_left, cell_size, color)
                            font = pygame.font.Font(None, 18)
                            text = font.render("ylop", True, (255, 255, 255))
                            shadow = font.render("ylop", True, (0, 0, 0))
                            text_rect = text.get_rect(center=(x * cell_size + cell_size // 2, y * cell_size + cell_size // 2 + 8))  # Slightly lower
                            shadow_rect = text_rect.copy()
                            shadow_rect.x += 1
                            shadow_rect.y += 1
                            self.screen.blit(shadow, shadow_rect)
                            self.screen.blit(text, text_rect)
                            
                        elif prop_type == "triangle":
                            pygame.draw.polygon(self.screen, color, [
                                (center[0], center[1] - cell_size // 4),  # Top
                                (center[0] - cell_size // 4, center[1] + cell_size // 4),  # Bottom-left
                                (center[0] + cell_size // 4, center[1] + cell_size // 4)   # Bottom-right
                            ])  # Triangle for triangles
                            count = parts[2]  
                            font = pygame.font.Font(None, 28)  # Larger font size
                            text = font.render(count, True, (255, 255, 255))  # White text
                            shadow = font.render(count, True, (0, 0, 0))
                            shadow_pos = (center[0] - 7 + 1, center[1] - 20 + 1)
                            self.screen.blit(shadow, shadow_pos)
                            text_pos = (center[0] - 7, center[1] - 20)
                            self.screen.blit(text, text_pos)
                        
                        elif prop_type == "square":
                            # Draw a filled square at the center of the cell
                            square_size = cell_size // 2
                            square_rect = pygame.Rect(
                                center[0] - square_size // 2,
                                center[1] - square_size // 2,
                                square_size,
                                square_size
                            )
                            pygame.draw.rect(self.screen, color, square_rect)
                            
                        elif prop_type == "dot":
                            pygame.draw.circle(self.screen, (0, 0, 0), center, cell_size // 8)  # Small black dot

        
        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS

        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def _draw_polyshape(self, surface, shape_array, top_left, cell_size, color):
        shape_height = len(shape_array)
        shape_width = len(shape_array[0])

        padding = cell_size // 6

        mini_block_width = (cell_size - 2 * padding) // shape_width
        mini_block_height = (cell_size - 2 * padding) // shape_height

        for y, row in enumerate(shape_array):
            for x, val in enumerate(row):
                if val:
                    rect = pygame.Rect(
                        top_left[0] + padding + x * mini_block_width,
                        top_left[1] + padding + y * mini_block_height,
                        mini_block_width,
                        mini_block_height
                    )
                    pygame.draw.rect(surface, color, rect)
    
    def _draw_star(self, surface, color, center, radius):
        # 5 points for a classic star
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            r = radius if i % 2 == 0 else radius // 2
            x = center[0] + int(math.cos(angle) * r)
            y = center[1] - int(math.sin(angle) * r)
            points.append((x, y))
        pygame.draw.polygon(surface, color, points)
        
    def _get_color_from_name(self, parts):
        """
        Helper function to extract color from property name parts.
        """
        if "red" in parts:
            return (255, 0, 0)  # Red
        elif "blue" in parts:
            return (0, 0, 255)  # Blue
        elif "yellow" in parts:
            return (255, 255, 0)  # Yellow
        elif "green" in parts:
            return (0, 255, 0)  # Green
        elif "black" in parts:
            return (0, 0, 0)  # Black
        elif "purple" in parts:
            return (128, 0, 128)  # Purple
        elif "orange" in parts:
            return (255, 165, 0)  # Orange
        elif "white" in parts:
            return (255, 255, 255)  # White
        else:
            return (128, 128, 128)  # Default: Gray
        
        

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
