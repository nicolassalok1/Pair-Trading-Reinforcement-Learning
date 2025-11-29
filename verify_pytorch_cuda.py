"""
Quick CUDA sanity check for PyTorch.
Run inside the target conda environment:
    python verify_pytorch_cuda.py
"""

import torch


def main() -> None:
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA runtime version: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("CUDA not available. Check driver/toolkit installation and that you're in the right conda env.")
        return

    device_index = 0
    device_name = torch.cuda.get_device_name(device_index)
    print(f"CUDA device {device_index}: {device_name}")

    # Minimal tensor test on GPU
    x = torch.randn(4, 4, device=f"cuda:{device_index}")
    y = torch.randn(4, 4, device=f"cuda:{device_index}")
    z = x @ y
    print(f"Tensor device: {z.device}, sum: {z.sum().item():.4f}")


if __name__ == "__main__":
    main()
