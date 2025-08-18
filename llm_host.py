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
import logging
import yaml
import asyncio
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd


load_dotenv()
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL").rsplit("/chat/completions",1)[0]
client = OpenAI(api_key=API_KEY, base_url=API_URL)

ds = load_dataset("lkaesberg/SPaRC", 'all', split="test")
df = ds.to_pandas()

def make_json_safe(obj, seen=None):
    if seen is None: seen=set()
    oid=id(obj)
    if oid in seen: return None
    seen.add(oid)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: make_json_safe(v,seen) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [make_json_safe(v,seen) for v in obj]
    if isinstance(obj,(int,float,str,bool)) or obj is None: return obj
    return str(obj)

async def run_episode(i):
    env = gym.make("env-SPaRC-v1", puzzles=df, render_mode=None, traceback=True, max_steps=100)

    logger = logging.getLogger(f"episode_{i}")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(f"logfiles_t/puzzle{i}.log", mode="w", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info(f"Episode {i+1}/{len(df)} start")

    for _ in range(i+1):
        obs, info = env.reset()

    reward = 0
    if i == len(df) - 1:
        polyshapes = df['polyshapes'][0]
    else:
        polyshapes = df['polyshapes'][i+1]
    
    polyshapes = yaml.safe_load(polyshapes)

    system_prompt = f"""
    You are an autonomous agent controlling a path‐finding puzzle solver.
    Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided grid, adhering to all puzzle rules.

    Core Concepts & Grid Basics:
    Grid Dimensions: You can find the puzzle grid size in the info 
    Path: The solution is a single, continuous line connecting adjacent nodes either horizontally or vertically.
    Revisiting: You can traceback your path, but you MUST do so in the same way you came, without crossing over your own path. 
    When tracing back, you can only move to the last cell you occupied, and then continue from there. Also when you trace back, the nodes you no longer use in your path are free to be used again.
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both x and y are odd. 
    The path goes around these rule cells, never on them. They are also marked as gaps.
    Regions: The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.

    Detailed Solving Rules:
    The drawn path must satisfy ALL applicable constraints:

    1.  Path Constraints:
        Path connects adjacent nodes (horizontal/vertical moves only).
        Nodes CAN be revisited. But only if you trace back to the last cell you occupied(And from there again and again ...).
        Otherwise you CANNOT cross your own path.
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

    Polyshape Definitions: Shapes are defined by 2D arrays where 1 indicates an occupied cell and 0 indicates an empty cell. {polyshapes}

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
    
    Keep_Turns = 4
    
    messages = [{"role":"system","content":system_prompt}]

    for step in range(101):
        user_payload = json.dumps(make_json_safe({'obs':obs,'info':info,'reward':reward}))
        messages.append({"role":"user","content":user_payload})

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="Qwen/Qwen3-32B",
            messages=messages,
            temperature=0.0
        )

        reply = response.choices[0].message.content.strip()
        last_line = reply.splitlines()[-1].strip()
        m = re.match(r"^(?:Final:\s*)?([0-3])$", last_line)
        if not m:
            logger.error(
                        "Puzzle %d: invalid model output, no “Final: <0–3>” found – skipping.\n%s",
                        i+1, reply
                    )
            logger.info("Puzzle %d aborted due to invalid output.", i+1)
            return  # bail out of this episode        
        action = int(m.group(1))

        obs, reward, terminated, truncated, info = env.step(action)

        # log everything
        logger.info(
            "Step %d | prompt_tokens=%d | completion_tokens=%d | total_tokens=%d | reward=%f | reply=%s",
            step,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
            reward,
            reply
        )
        
        messages.append({"role":"assistant","content": f"Final: {action}"})
        
        system = messages[0]
        tail = messages[-(Keep_Turns*2):]
        messages = [system] + tail
        
        if terminated or truncated:
            logger.info("Puzzle %d difficulty: %d", i+1, info["difficulty"])
            break
        
    if terminated:
        logger.info(
            "Episode %d terminated after %d steps; final reward=%f ; difficulty=%d",
            i+1, step+1, reward, info["difficulty"]
        )
    elif truncated:
        logger.info(
            "Episode %d truncated after %d steps; final reward=%f ; difficulty=%d",
            i+1, step+1, reward, info["difficulty"]
        )
    
    logger.info("Episode %d done", i+1)

async def main():
    tasks = [run_episode(i) for i in range(len(df))]
    for finished in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Episodes"
    ):
        await finished

if __name__=="__main__":
    asyncio.run(main())
