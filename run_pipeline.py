import subprocess
import sys
import os

def run_command(command, description):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    # This runs the command in the shell (compatible with Windows/Linux)
    try:
        # sys.executable ensures we use the same python interpreter (virtual env)
        full_command = [sys.executable] + command
        subprocess.run(full_command, check=True)
        print(f">>> {description} COMPLETED SUCCESSFULLY.")
    except subprocess.CalledProcessError as e:
        print(f">>> ERROR in {description}.")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure results folder exists
    os.makedirs('results', exist_ok=True)

    # 1. Train BAN
    run_command(['src/train.py', '--model', 'ban'], "Training BAN Model")

    # 2. Train BLIP-2
    run_command(['src/train.py', '--model', 'blip2'], "Training BLIP-2 Model")

    # 3. Compare Models
    run_command([
        'src/compare_models.py', 
        '--ban_checkpoint', 'results/ban_model.pth',
        '--blip_checkpoint', 'results/blip2_model.pth'
    ], "Comparing Models & Generating Report")