Param(
    [string]$EnvName = "rl-pytorch-cuda"
)

$ErrorActionPreference = "Stop"

function Ensure-Conda {
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        return
    }
    throw "conda not found in PATH. Open an Anaconda/Miniconda PowerShell prompt or add conda to PATH."
}

function Env-Exists($name) {
    conda env list | Select-String "^\s*$name\s" | ForEach-Object { return $true }
    return $false
}

function Create-Or-Update-Env {
    param(
        [string]$name
    )

    if (Env-Exists $name) {
        Write-Host "Environment '$name' already exists. Updating from environment.yml..." -ForegroundColor Yellow
        conda env update -n $name -f environment.yml
    }
    else {
        Write-Host "Creating environment '$name' from environment.yml..." -ForegroundColor Green
        conda env create -n $name -f environment.yml
    }
}

function Activate-Env {
    param(
        [string]$name
    )

    conda activate $name
}

function Run-Verification {
    python verify_pytorch_cuda.py
}

# Main
Ensure-Conda
Create-Or-Update-Env -name $EnvName
Activate-Env -name $EnvName
Run-Verification
Activate-Env -name $EnvName
Write-Host "Updating conda (base) from defaults..." -ForegroundColor Cyan
conda update -n base -c defaults conda -y
