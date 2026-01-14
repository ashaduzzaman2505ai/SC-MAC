import sys
import os

# Ensure the 'src' directory is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.worker import LogicWorker

def main():
    # Instantiate Agent
    agent = LogicWorker()
    
    # Example Problem: Logical Syllogism
    # Pre-load the verifier with a fact for the test
    agent.verifier.add_premise("Fact: IsBird")
    
    problem = "If it is a bird (IsBird), and all birds fly, can we conclude it does NOT fly?"
    
    print("\n--- Starting SC-MAC Experiment ---")
    result = agent.run_sc_mac_loop(problem)
    
    print("\n--- FINAL PAPER RESULT ---")
    print(result)

if __name__ == "__main__":
    main()