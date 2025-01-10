import os
import json
import pandas as pd

# Function to recursively find all JSON files in the directory
def find_json_files(directory, filename="graphmeasures.json"):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == filename:
                json_files.append(os.path.join(root, file))
    return json_files

# Function to load the content of the JSON file and return it as a dict
def load_json_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Function to check if the CSV file already exists and has content
def csv_exists_and_nonempty(csv_path):
    return os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

# Main function to iterate through the files and append data to an existing CSV
def append_json_to_csv(directory, output_csv):
    json_files = find_json_files(directory)
    data_list = []

    for json_file in json_files:
        try:
            json_data = load_json_data(json_file)
            json_data['file_path'] = json_file  # Add the file path to the data
            data_list.append(json_data)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    if data_list:  # If there are new entries
        df = pd.DataFrame(data_list)

        # Check if the CSV file already exists
        if csv_exists_and_nonempty(output_csv):
            # Append to existing CSV without writing the header again
            df.to_csv(output_csv, mode='a', header=False, index=False)
        else:
            # Create a new CSV with headers
            df.to_csv(output_csv, mode='w', header=True, index=False)

import sys
# Specify your directory and output CSV file
directory = sys.argv[1]
output_csv = sys.argv[2]
# Call the function
append_json_to_csv(directory, output_csv)
