import os
import subprocess

os.makedirs("results", exist_ok=True)
files = [
    "training.train_curriculum",
]

with open("results/phase3_result.txt", "w", encoding="utf-8") as f:
    f.write("Phase 3 - Curriculum Training\n")
    f.write("Files created: training/train_curriculum.py\n\n")
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
            
    f.write("\nNext phase: Phase 4 - Evaluation & Baselines\n")
print("Done writing phase3_result.txt")
