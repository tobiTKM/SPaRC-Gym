# Gym Environment for SpaRC

----
## Description
A custom Gymnasium environment for SPaRC. The Game and Dataset was develop by: https://sparc.gipplab.org/ . This project allows LLM agents and humans to interact/play the puzzles used in SPaRC. For how the puzzles work also look up https://sparc.gipplab.org/ .

## Installation

[📦 PyPI Package](https://pypi.org/project/Gym-Env-SPaRC/)

Install the package from PyPI:

```bash
pip install Gym-Env-SPaRC
```

Or install from source:

```bash
git clone https://github.com/tobiTKM/Gym-Environment_for_SPaRC.git
cd Gym-Environment_for_SPaRC
pip install -e .
```

## Quick Start

To create the Gym Environment:

```python
import gymnasium as gym
import gymnasium_env_for_SPaRC
env = gym.make("env-SPaRC-v1", puzzles=df, render_mode='human', observation='new',traceback=True, max_steps=1000)
```

### Options

| Option        | Default | Options                |Description                        |
|:--------------|:-------:|:----------------------:|-----------------------------------|
| puzzles       | required| pd.Dataframe           | Pandas DataFrame of SPaRC puzzles |
| render_mode   |  None   | 'human', 'llm', or None| Which Visualization to use        |
| observation   |  'new'  | 'new' or 'SPaRC'       | Which Observation type to use     |
| traceback     |  False  | False or True          | Allow the agent to backtrack      |
| max_steps     |  2000   | any int                | Maximum steps per episode         |


## Core Functions

```python
env.reset() -> Observation, Info: dict
```
Resets the Environment, moves to the next puzzle. Returns the Initial Observation and Info.

```python
env.step(action: int(0-3)) -> observation, reward: int, terminated: bool, truncated: bool, info: dict
```
Moves the Environment one Step based on the Action. Returns the new Observation, Info, Reward and if the puzzle is finished.

```python
env.render()
```
Visualizes the Puzzle's current State

```python
env.close()
```
Close the environment and cleanup any resources.


## Environment Details

#### Action Space

- **Discrete(4)**: Represents the four possible moves:
    - **0**: Right
    - **1**: Up
    - **2**: Left
    - **3**: Down

#### Observation Space

if Observation = 'new':

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

if Observation = 'SPaRC':

- **Json String**
Same Representation as in SPaRC https://github.com/lkaesberg/SPaRC .
Json String Representation of a 2D array of the grid capturing all different properties as Symbols (Strings).


#### Reward System

- **Sparse Rewards**:
  - `+1`: For solving the puzzle.
  - `-1`: For Failing.
  - `+0.01`: For staying on a solution path on each step.


## Folder Structure

- Gym-Environment_for_SPaRC/ # Custom environment implementation
    - gymnasium_env_for_SPaRC/ # Core environment logic
        - init.py # Environment initialization 
        - gym_env_for_SPaRC.py # Core environment logic 
        - register_env.py # Environment registration 
    - llm_testing/
      - llm_host.py # Example script for using the gym with a llm 
        - parse_logs.py # Script to filter out the results of the created logfiles from llm_host.py 
    - Final_Product.py # Main script for human interaction
    - human_play.py # Helper Function for human play 
    - pyproject.toml
    - LICENCE
    - README.md


## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.


## LICENSE

This project is licensed under the MIT License - see the LICENCE file for details.


----

## Acknowlegdments

Special thanks to Lars Benedikt Kaesberg (l.kaesberg@uni-goettingen.de) and Jan Philip Wahle for giving me the opportunity to do this Project aswell as supervising the Project.