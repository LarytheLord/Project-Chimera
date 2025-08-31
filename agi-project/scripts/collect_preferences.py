
import os
import sys
import json

# Add the project root to the Python path to allow imports from 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.cognitive_core.prometheus_core import PrometheusCognitiveCore
from src.agent.tool_user import ToolRegistry, WebSearchTool
from src.agent.agent import Agent

def main():
    """Main function to run the interactive preference collection script."""
    print("--- Starting Preference Collection Script ---")
    print("Your input will be used to generate two responses from the agent.")
    print("You will then choose the better response to help train the agent.")
    print("Type 'quit' or 'exit' to stop.")

    # 1. Initialize the Agent
    try:
        # This setup is taken from main.py
        cognitive_core = PrometheusCognitiveCore(
            api_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        )
        cognitive_core.load_model("gemini-pro")
        
        tool_registry = ToolRegistry()
        tool_registry.register_tool(WebSearchTool())
        
        # The agent itself isn't strictly needed, we just need its cognitive core
        # but initializing it is a good way to ensure all components are available.
        db_path = os.path.join(project_root, "memory_db")
        agent = Agent(cognitive_core=cognitive_core, tool_registry=tool_registry, db_path=db_path)

    except ValueError as e:
        print(f"\n--- CONFIGURATION ERROR ---")
        print(f"Error initializing the agent: {e}")
        print("Please make sure the GEMINI_API_KEY environment variable is set correctly.")
        print("---")
        return

    output_file = os.path.join(project_root, "preference_data.jsonl")
    print(f"Preferences will be saved to: {output_file}\n")

    # 2. Main collection loop
    while True:
        try:
            prompt = input("Enter a prompt for the agent > ")
            if prompt.lower() in ['quit', 'exit']:
                break

            if not prompt:
                continue

            print("\nGenerating responses...")
            # Generate two responses with different temperatures for variety
            response_a = agent.cognitive_core.generate_response({"text_data": prompt}, temperature=0.6)
            response_b = agent.cognitive_core.generate_response({"text_data": prompt}, temperature=1.0)

            print("--- Please choose the better response ---")
            print(f"\n[A]: {response_a}")
            print(f"\n[B]: {response_b}\n")

            # 3. Get user preference
            choice = ''
            while choice.upper() not in ['A', 'B', 'SKIP']:
                choice = input("Which response is better? (A/B/skip) > ")
            
            if choice.upper() == 'SKIP':
                print("Skipping preference.\n")
                continue

            # 4. Save the preference data
            chosen = response_a if choice.upper() == 'A' else response_b
            rejected = response_b if choice.upper() == 'A' else response_a

            preference_data = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            }

            with open(output_file, 'a') as f:
                f.write(json.dumps(preference_data) + '\n')
            
            print("Preference saved!\n")

        except KeyboardInterrupt:
            print("\nStopping script.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

    print("--- Preference Collection Script Finished ---")

if __name__ == "__main__":
    main()
