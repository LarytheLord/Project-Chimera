
import os
import sys
import json

def verify_data():
    """Reads the preference data file and verifies that each line is valid JSON."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    preference_file = os.path.join(project_root, "preference_data.jsonl")
    
    print(f"Verifying data file: {preference_file}")
    
    try:
        with open(preference_file, 'r') as f:
            for i, line in enumerate(f):
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error on line {i+1}: {e}")
                    print(f"Invalid line: {line.strip()}")
                    sys.exit(1)
        
        print("Data verification successful. All lines are valid JSON.")
        sys.exit(0)

    except FileNotFoundError:
        print(f"Error: Preference data file not found at {preference_file}")
        sys.exit(1)

if __name__ == "__main__":
    verify_data()
