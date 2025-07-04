# Gym Environment for SpaRC Project

----
## Description
A custom Gymnasium environment for SPaRC. The Game and Dataset was develop by: https://sparc.gipplab.org/ . This project allows LLM agents and humans to interact/play the puzzles used in SPaRC. For how the puzzles work also look up https://sparc.gipplab.org/ .

----
## Arguments when creating the Gym Env:

- **puzzles(pd.DataFrame)** The puzzles that should be used. The puzzles must come in the shape of https://sparc.gipplab.org/ . 
- **render_mode(str)** Optional, If render_mode='human' the Gym Environment will visualize every step using `pygame`. If no argumentpassed: no render mode activated.
- **traceback (bool)** When set to True it allows the Agent to move back on his path, if False it does not. If argument not passed: traceback=False. 
- **max_steps(int)** Optional, the maximum amount of steps the Gym environment will runbefore it terminates. If no argument passed: max_steps=200.

----
## Installation and Usage
how to use:
- **Either:**
  - Run pip install Gym-Env-SPaRC
  - import gymnasium_env_for_SPaRC (and gymnasium)
  - make the gym using: env = gym.make("env-SPaRC-v0", puzzles=df, render_mode='human', traceback=False, max_steps=max_steps) (example)
  - and use the gym to your liking, examples how to use are in Final_Product.py or llm_host.py
- **or:**
  - clone the repository
  - install the dependecies
  - Run Final_Product.py to play as a human or customize Final_Product.py or llm_host.py to your liking

#### Packages with Versions:
- gymnasium>=0.28.1
- numpy>=1.26.4
- pygame>=2.2.0
- pyyaml>=5.1
- pandas>=2.2.1
- huggingface-hub>=0.15.0
- fsspec>=2024.1.1
- chardet>=5.2.0

----
## Environment Details

#### Action Space

- **Discrete(4)**: Represents the four possible moves:
    - **0**: Right
    - **1**: Up
    - **2**: Left
    - **3**: Down

#### Observation Space

- **Dict** A Dictionary of:
  - **base: Dict**: A dictionary of 2D arrays representing the puzzle state:
    - `"visited"`: Tracks visited cells.
    - `"gaps"`: Represents gaps in the grid.
    - `"agent_location"`: Current position of the agent.
    - `"target_location"`: Goal position.
    - Additional keys for unique properties like `"stars"`, `"triangles"`, etc.
    - The 2D Arrays are of the shape of the puzzle and are One-hot Encoded
  - **color: list**: A 2D Array representing the colors of the properties.
    - 8 possible colors are represented with 1-8
    - 2D Array is of shape of the puzzle
  - **additional_info: list** A 2D Array with additional Info about the puzzle
    - 2D Array is of shape of the puzzle
    - Possible additional_info:
    - ID of the polyshape
    - Count of the Triangles


#### Reward System

- **Sparse Rewards**:
  - **Outcome Reward**:
    - `+1`: For solving the puzzle.
    - `0`: For intermediate steps.
    - `-1`: for Failing.
  - **Normal Reward**:
    - `+1`: For solving the puzzle.
    - `-1`: For Failing.
    - `+0.01`: For staying on a solution path on each step.

- **Note:** The Reward from the step function is the **Normal Reward**, **Outcome Reward** can be found in the Info if needed.


## Visualization

The environment uses `pygame` for rendering:

- **Agent**: Blue square.
- **Target**: Red square.
- **Gaps**: Dark Green cells.
- **Visited cells**: light Green cells.
- **Unique Properties**:
  - **Stars**: colored star.
  - **Square**: colored square.
  - **Triangles**: Colored triangles with counts.
  - **Polyshapes**: Colored polygons.
  - **Ylop**: Colored Polygons with marker ylop.
  - **Dots**: Small black circles.

## Folder Structure

- Gym-Environment_for_SPaRC/ # Custom environment implementation
    - gymnasium_env/ # Core environment logic
        - init.py # Environment initialization 
        - gym_env_for_SPaRC.py # Core environment logic 
        - register_env.py # Environment registration 
    - Final_Product.py # Main script for human interaction
    - llm_host.py # Example script for using the gym with a llm
    - human_play.py # Function for human play 
    - parse_logs.py # Script to filter out the results of the created logfiles from llm_host.py 
    - README.md # Project documentation

----

## Acknowlegdments

Special thanks to Lars Benedikt Kaesberg (l.kaesberg@uni-goettingen.de) and Jan Philip Wahle for giving me the opportunity to do this Project aswell as supervising the Project.