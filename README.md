# TheWitnessGame Gym Project

----
## Description
A custom Gymnasium environment for the Game TheWitness. The Game and Dataset was develop by: https://sparc.gipplab.org/ . This project allows reinforcement learning (RL) agents and humans to interact with the Game. For how the game works also look up https://sparc.gipplab.org/ .

----
## Features
- **Processes Puzzles** Loads and processes puzzles of shape of this dataset https://sparc.gipplab.org/.
- **Human Interaction**: Play the puzzles interactively using arrow keys.
- **Visualization**: Visualize the current states of the puzzles using `pygame`.
- **Legal Action Enforcement**: Only valid moves are allowed, ensuring agents and humans adhere to puzzle rules.
- **Dynamic Observations**: Observation space includes 2D arrays for each property, such as visited cells, gaps, and unique puzzle elements.

----
## Installation and Usage
how to use:
- clone the repository
- install the dependecies
- Run Final_Product.py to play as a human or customize Final_product.py how you want to use the gym

#### Packages with Versions:
- gymnasium=0.28.1
- numpy=1.26.4
- pygame=2.2.0
- yaml=0.2.5
- pandas=2.2.1

----
## Environment Details

#### Action Space

- **Discrete(4)**: Represents the four possible moves:
    - **0**: Right
    - **1**: Up
    - **2**: Left
    - **3**: Down

#### Observation Space

- **Dict**: A dictionary of 2D arrays representing the puzzle state:
  - `"visited"`: Tracks visited cells.
  - `"gaps"`: Represents gaps in the grid.
  - `"agent_location"`: Current position of the agent.
  - `"target_location"`: Goal position.
  - Additional keys for unique properties like `"star_red"`, `"triangle_blue_3"`, etc.

#### Reward System

- **Sparse Rewards**:
  - `+1`: For solving the puzzle.
  - `0`: For intermediate steps and failing.

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

- Gym-TheWitnessGame/ # Custom environment implementation
    - gymnasium_env/ # Core environment logic
        - init.py # Environment initialization 
        - gym_Witness.py # Core environment logic 
         - register_env.py # Environment registration 
    - Final_Product.py # Main script for human interaction
    - human_play.py # Function for human play 
    - README.md # Project documentation

----

## Acknowlegdments

Special thanks to Lars Benedikt Kaesberg (l.kaesberg@uni-goettingen.de) and Jan Philip Wahle for giving me the opportunity to do this Project aswell as supervising the Project.