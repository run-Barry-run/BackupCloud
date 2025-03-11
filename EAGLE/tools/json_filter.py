import json

def filter_json_by_key(input_file, output_file, key):
    """
    Filters a JSON file to create a new JSON file containing only dictionaries that have the specified key.

    :param input_file: Path to the input JSON file.
    :param output_file: Path to the output JSON file.
    :param key: The key to filter dictionaries by.
    """
    try:
        # Load the input JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Filter dictionaries that contain the specified key
        filtered_data = [item for item in data if key in item.keys()]

        output_file = output_file + f'eagle-{len(filtered_data)}-new.json'
        # Save the filtered data to the output JSON file
        with open(output_file, 'w') as f:
            json.dump(filtered_data, f)

        print(f"Filtered JSON saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = 'dataset/Eagle-1.8M/eagle-1-sft-1_8M.json'  # Path to the input JSON file
output_file = 'dataset/Eagle-1.8M/'  # Path to the output JSON file
key = 'image'  # Key to filter by

filter_json_by_key(input_file, output_file, key)