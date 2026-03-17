import json
import sys
from pathlib import Path

def verify_data():
    """Reads the preference data file and verifies that each line is valid JSON."""
    preference_file = Path(__file__).resolve().parents[1] / "preference_data.jsonl"
    
    print(f"Verifying data file: {preference_file}")
    
    try:
        with preference_file.open('r', encoding='utf-8') as f:
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
