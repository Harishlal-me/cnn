import torch
import sys

def verify_gpu():
    print("Checking CUDA availability...")
    if not torch.cuda.is_available():
        print("\nERROR: CUDA GPU not detected. Training cannot continue.")
        print("Suggested fixes:")
        print("  * install CUDA-enabled PyTorch")
        print("  * check NVIDIA drivers")
        print("  * run nvidia-smi")
        sys.exit(1)
        
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("CUDA available: True")
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram_gb:.1f} GB")

if __name__ == "__main__":
    verify_gpu()
