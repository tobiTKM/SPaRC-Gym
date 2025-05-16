import pandas as pd
import numpy as np
import yaml

def process_puzzles(df):
    """
    Processes a DataFrame of puzzles and returns a list of puzzle dictionaries.

    Parameters:
        df (pd.DataFrame): The DataFrame containing puzzle data.

    Returns:
        list: A list of dictionaries, each representing a processed puzzle.
    """
    
    if df is None:
            raise ValueError("No dataframe provided")
        
    puzzles = []

    for i in range(len(df)):

        puzzle = {}
        
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
        puzzle.update({'solution_count': solution_count, 'solution_paths': solution_paths[0]})
        
        # Extract start and target locations
        text_visualization = df['text_visualization'][i]
        text_yaml = yaml.safe_load(text_visualization)
        start_location = (text_yaml["puzzle"]["start"]["x"], text_yaml["puzzle"]["start"]["y"])
        target_location = (text_yaml["puzzle"]["end"]["x"], text_yaml["puzzle"]["end"]["y"])
        puzzle.update({'start_location': start_location, 'target_location': target_location})
        
        # Initialize observation arrays
        obs_array = {
            'visited': np.zeros((y_size, x_size), dtype=int),
            'gaps': np.zeros((y_size, x_size), dtype=int),
            'agent_location': np.zeros((y_size, x_size), dtype=int),
            'target_location': np.zeros((y_size, x_size), dtype=int)
        }
        
        # Extract unique properties
        unique_properties = set()
        for cell in text_yaml["puzzle"]["cells"]:
            properties = cell.get("properties", {})
            for key, value in properties.items():
                if key == 'type':
                    if value == 'star' or value == 'square':
                        combined = f"{value}_{properties.get('color', '')}"
                    elif value == 'triangle':
                        combined = f"{value}_{properties.get('color', '')}_{properties.get('count', '')}"
                    else:
                        combined = f"{value}_{properties.get('polyshape', '')}_{properties.get('color', '')}"
                    unique_properties.add(combined)
                    
                elif key == 'dot':
                    combined = 'dot'
                # Add new property to obs_array if not already present
                if combined not in obs_array:
                    obs_array.update({combined: np.zeros((y_size, x_size), dtype=int)})
        
        unique_property_count = len(unique_properties) + 4  # Adding 4 for the base properties
        puzzle.update({'unique_properties': unique_property_count})
        
        # Populate observation arrays
        for cell in text_yaml["puzzle"]["cells"]:
            position = cell.get("position", {})
            properties = cell.get("properties", {})
            x, y = position.get("x"), position.get("y")

            for key, value in properties.items():
                if key == 'type':
                    if value == 'star' or value == 'square':
                        combined = f"{value}_{properties.get('color', '')}"
                    elif value == 'triangle':
                        combined = f"{value}_{properties.get('color', '')}_{properties.get('count', '')}"
                    else:
                        combined = f"{value}_{properties.get('polyshape', '')}_{properties.get('color', '')}"
                
                # Update the corresponding observation array
                if combined in obs_array:
                    obs_array[combined][y, x] = 1

        puzzle.update({'obs_array': obs_array})
        
        # Add the processed puzzle to the list
        puzzles.append(puzzle)
    
    return puzzles

# Example usage
#splits = {'train': 'puzzle_all_train.jsonl', 'test': 'puzzle_all_test.jsonl'}
#df = pd.read_json("hf://datasets/lkaesberg/SPaRC/" + splits["train"], lines=True)

