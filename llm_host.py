import json
import gymnasium as gym
from openai import OpenAI
import pandas as pd
import gymnasium_env_for_SPaRC
from dotenv import load_dotenv
import os

'''
This script demonstrates how to use the GYm environement for SPaRC with an LLM.
It initializes the environment, sets up the LLM client, and runs a loop to interact with
the environment using the LLM to decide actions based on observations.
'''

# Load the environment variables from the .env file
load_dotenv()              
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

client = OpenAI(
    api_key=f"{API_KEY}",
    base_url=f"{API_URL}",
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
At each turn you'll receive the current observation as JSON.
Respond with exactly one integer: 0=right, 1=up, 2=left, 3=down.
"""

# Initialize the messages list with the system prompt
messages = [
    {"role": "system", "content": system_prompt}
]

# Main loop to interact with the environment
for step in range(max_steps):
    user_content = {
        'observation': obs,
        'info': info,
        'reward': reward
    }
    messages.append({
        "role": "user",
        "content": json.dumps(user_content)
    })

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        temperature=0.0,
        max_tokens=4
    )
    reply = response.choices[0].message.content.strip()
    
    action = int(reply)

    obs, reward, terminated, truncated, info = env.step(action)

    messages.append({"role": "assistant", "content": reply})

    if terminated or truncated:
        print(f"Episode ended,  reward={reward}, info={info}")
        break
