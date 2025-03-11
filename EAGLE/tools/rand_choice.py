import json
import random

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def choose_random_dicts(data, count):
    return random.sample(data, count)

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    input_file_path = 'dataset/3D-LLM/data_part2_scene.json'
    output_file_path = 'dataset/3D-LLM/100_part2_scene.json'
    
    data = load_json(input_file_path)
    random_dicts = choose_random_dicts(data, 100)
    save_json(random_dicts, output_file_path)