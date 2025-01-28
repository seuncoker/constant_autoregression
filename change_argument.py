import json
from typing import List

import argparse

# Define an argument parser
parser = argparse.ArgumentParser(description="Your script description")

# Add arguments with options and help text
parser.add_argument("--argument_loc", type=str, help=" argument name")
parser.add_argument("--key", type=str, help=" key")
parser.add_argument("--new_value", type=str, help="new_value")


# Parse arguments from command line
args = parser.parse_args()

# Access arguments using their names
argument_loc = args.argument_loc
key = args.key
new_value = args.new_value

# Print arguments for demonstration
print(f"argument_loc: {argument_loc}")
print(f"key: {key}")
print(f"new_value: {new_value}")
print("\n")



def change_json_value(argument_loc, key, new_value):
  """
  This function reads a JSON file, changes the value of a specific key, and writes the updated data back to the file.

  Args:
      filename (str): The name of the JSON file.
      key (str): The key whose value you want to change.
      new_value (any): The new value to assign to the key.
  """
  with open(argument_loc, 'r') as f:
   data = json.load(f)

  # Check if the key exists before modification
  if key in data:
    data[key] = new_value
    print(new_value)
  else:
   print(f"Warning: Key '{key}' not found in JSON file.")

  print("changing values in the arguments_test.json")
  print(f"arguments_test.json location:  {argument_loc}")

  with open(argument_loc, 'w') as f:
     json.dump(data, f, indent=2)  # indent for readability (optional)
  print("saving arguments_test.json")
  
# Example usage



change_json_value(argument_loc, key, new_value)
  
  