powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
conda activate rl-pytorch-cuda
python verify_pytorch_cuda.py
