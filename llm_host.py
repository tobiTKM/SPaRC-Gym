import json
import gymnasium as gym
from openai import OpenAI
import pandas as pd
import gymnasium_env_for_SPaRC
from dotenv import load_dotenv
import os
import numpy as np   
import os
import re

'''
This script demonstrates how to use the Gym environement for SPaRC with an LLM.
It initializes the environment, sets up the LLM client, and runs a loop to interact with
the environment using the LLM to decide actions based on observations.
'''

# Load the environment variables from the .env file
load_dotenv()              
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")
API_URL = API_URL.rsplit("/chat/completions", 1)[0]

# Initialize the OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL,
)

# Load the SPaRC dataset
splits = {'train': 'puzzle_all_train.jsonl', 'test': 'puzzle_all_test.jsonl'}
df = pd.read_json("hf://datasets/lkaesberg/SPaRC/" + splits["train"], lines=True)

# Initialize the SPaRC environment
max_steps = 1000  # Define the maximum number of steps for the episode
env = gym.make("env-SPaRC-v0", puzzles=df, render_mode='human', traceback=False, max_steps=max_steps)
obs, info = env.reset()
reward = 0


# Define the system prompt for the LLM
system_prompt = """
You are an autonomous agent controlling a path‐finding puzzle solver.
Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided grid, adhering to all puzzle rules.

Core Concepts & Grid Basics:
Grid Dimensions: You can find the puzzle grid size in the info 
Path: The solution is a single, continuous line connecting adjacent nodes either horizontally or vertically.
No Revisits: The path CANNOT visit the same node more than once.
Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both x and y are odd. 
The path goes around these rule cells, never on them. They are also marked as gaps.
Regions: The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.

Detailed Solving Rules:
The drawn path must satisfy ALL applicable constraints:

1.  Path Constraints:
    Path connects adjacent nodes (horizontal/vertical moves only).
    Nodes CANNOT be revisited.
    Path MUST pass through all Dot cells.
    Path CANNOT pass through any Gap cells.

2.  Region-Based Rules (Apply to areas enclosed by the path):
    Squares: All squares within a single region MUST be the same color. Squares of different colors MUST be separated into different regions by the path.
    Stars: Within a single region, each star symbol MUST be paired with exactly ONE other element of the same color. Other colors within the region are irrelevant to this specific star's rule.
    
    Polyshapes(poly): The region containing this symbol MUST be able to contain the specified shape (defined in Polyshape Definitions). The shape must fit entirely within the region's boundaries.
    If multiple positive polyshapes are in one region, the region must accommodate their combined, non-overlapping forms. Rotation of polyshapes is generally allowed unless context implies otherwise.
    
    Negative Polyshapes(ylop): These subtract shape requirements, typically within the same region as corresponding positive polyshapes.
    A negative polyshape cancels out a positive polyshape of the exact same shape and color within that region.
    If all positive shapes are canceled, the region has no shape constraint. A negative shape is only considered 'used' if it cancels a positive one.
    Negative shapes can sometimes rationalize apparent overlaps or boundary violations of positive shapes if interpreted as cancellations.

3.  Path-Based Rules (Edge Touching):
    Triangles: The path MUST touch a specific number of edges of the cell containing the triangle symbol.
        (1): Path touches EXACTLY 1 edge of the triangle's cell.
        (2): Path touches EXACTLY 2 edges of the triangle's cell.
        (3): Path touches EXACTLY 3 edges of the triangle's cell.
        (4): Path touches EXACTLY 4 edges (fully surrounds) the triangle's cell.

At each turn you'll receive current Information as JSON.
Observation: The current state of the grid, including the path and any rule cells.
The Observation is a dictionary of three elements:
base: dictionary of one-hot encoded 2D arrays of the grid, where a 1 indicates a cell is occupied by the key (eg. 'gap', 'dot', 'square', etc.).
color: 2D array of the grid marking the color of the property of each cell, where 0 indicates no color.
The colors are defined as follows: 1:red, 2:blue, 3:yellow, 4:green, 5:black, 6:purple, 7:orange, 8:white
additional_info: a 2D array of the grid containing additional information about the cells, such as the amount of edges for triangles, or the shape of polyshapes.

Info: Additional information about the puzzle, including:
solution_count: The number of valid solutions for the current puzzle.
difficulty: The difficulty level of the puzzle, ranging from 1 (easy) to 5 (hard).
grid_x_size: The width of the grid.
grid_y_size: The height of the grid.
legal_actions: A list of legal actions you can take, represented as integers (0=right, 1=up, 2=left, 3=down).
current_step: The current step number in the episode.
agent_location: The current location of the agent in the grid.
Rewards: A dictionary containing the normal reward and the outcome reward at the current step.

Reward: The current reward.

You MAY think step‐by‐step (feel free to “<think>…”), but you MUST end with:
Final: <digit>
where <digit> is exactly one of 0=right, 1=up, 2=left, 3=down.
No other output beyond your reasoning and that Final line.
"""

# Initialize the messages list with the system prompt
messages = [
    {"role": "system", "content": system_prompt}
]


def make_json_safe(obj, seen=None):
    """
    Recursively convert obj into JSON‐safe primitives.
    Drop circular refs by returning None.
    """
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return None
    seen.add(oid)

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_safe(v, seen) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v, seen) for v in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)

# Main loop to interact with the environment
for step in range(max_steps):
    user_content = {
        'observation': obs,
        'info': info,
        'reward': reward
    }
    
    safe_payload = make_json_safe(user_content)
    
    messages.append({
        "role": "user",
        "content": json.dumps(safe_payload)
    })

    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=messages,
        temperature=0.0,
    )
    
    reply = response.choices[0].message.content.strip()
    m = re.search(r"Final:\s*([0-3])", reply)
    if not m:
        raise ValueError(f"Could not find Final: <digit> in model output:\n{reply}")
    
    action = int(m.group(1))
    print(f"Step {step}: Action={action}")
    
    obs, reward, terminated, truncated, info = env.step(action)

    messages.append({"role": "assistant", "content": reply})

    if terminated or truncated:
        print(f"Episode ended,  reward={reward}, info={info}")
        break
