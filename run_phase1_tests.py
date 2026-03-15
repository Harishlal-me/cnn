import os
import subprocess

os.makedirs("results", exist_ok=True)
files = [
    "models.multiscale_cnn",
    "models.transformer_branch",
    "models.token_gate",
    "models.task_gate",
    "models.image_encoder",
    "models.cross_modal_gate",
    "models.task_heads",
    "models.mm_taskgate"
]

with open("results/phase1_result.txt", "w") as f:
    f.write("Phase 1 - Build the Model\n")
    f.write("Files created: " + ", ".join(f.replace(".", "/") + ".py" for f in files) + "\n\n")
    f.write("Tests executed & Outputs:\n")
    
    for module in files:
        f.write(f"\n--- Running {module} ---\n")
        try:
            result = subprocess.run(["python", "-m", module], capture_output=True, text=True, timeout=60)
            f.write(result.stdout)
            if result.stderr:
                f.write(result.stderr)
        except Exception as e:
            f.write(f"Failed: {str(e)}\n")
            
    f.write("\nNext phase: Phase 2 - Build the Training Pipeline\n")
print("Done writing phase1_result.txt")
