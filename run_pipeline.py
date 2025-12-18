import subprocess
import sys

def run_step(command, step_name):
    print(f"\n{'='*50}")
    print(f"Running Step: {step_name}")
    print(f"Command: {command}")
    print(f"{'='*50}\n")
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError:
        print(f"\n!!! Step '{step_name}' Failed. Aborting pipeline. !!!")
        sys.exit(1)

def main():
    # 1. Search Agent
    run_step("venv\\Scripts\\python src\\search_agent.py", "1. Search Agent (Download & Draft)")
    
    # 2. Refiner Agent
    run_step("venv\\Scripts\\python src\\refiner_agent.py", "2. Refiner Agent (Analysis & Auto-Publish)")
    
    # 3. Publisher Agent
    run_step("venv\\Scripts\\python src\\publisher_agent.py", "3. Publisher Agent (Build & Deploy)")

    print(f"\n{'='*50}")
    print(">>> All Steps Completed Successfully! <<<")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
