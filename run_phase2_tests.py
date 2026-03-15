import os
import subprocess

os.makedirs("results", exist_ok=True)
files = [
    "training.losses",
    "training.evaluate",
    "training.train"
]

with open("results/phase2_result.txt", "w", encoding="utf-8") as f:
    f.write("Phase 2 - Build the Training Pipeline\n")
    f.write("Files created: training/losses.py, training/train.py, training/evaluate.py, config.yaml\n\n")
    f.write("Tests executed & Outputs:\n")
    
    for module in files:
        f.write(f"\n--- Running {module} ---\n")
        try:
            result = subprocess.run(["python", "-m", module], capture_output=True, text=True, timeout=120)
            f.write(result.stdout)
            if result.stderr:
                f.write(result.stderr)
        except Exception as e:
            f.write(f"Failed: {str(e)}\n")
            
    f.write("\nNext phase: Phase 3 - Curriculum Training\n")
print("Done writing phase2_result.txt")
