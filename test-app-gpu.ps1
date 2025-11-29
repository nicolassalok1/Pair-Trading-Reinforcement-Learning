Param(
    [string]$PythonExe = "python",
    [switch]$SkipInstall,
    [switch]$UseCurrentEnv
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $repoRoot

Write-Host "Repository root: $repoRoot"
Write-Host "Using Python executable: $PythonExe"

# Use current environment or create/activate the project venv-gpu
if ($UseCurrentEnv) {
    Write-Host "Using current environment (aucune activation de venv-gpu)"
} else {
    $venvPath = Join-Path $repoRoot "venv-gpu"
    $activatePath = Join-Path $venvPath "Scripts\Activate.ps1"
    if (-not (Test-Path $activatePath)) {
        Write-Host "Creating GPU virtual environment at $venvPath"
        & $PythonExe -m venv $venvPath
    }

    Write-Host "Activating GPU virtual environment"
    . $activatePath
}

if (-not $SkipInstall) {
    Write-Host "Installing dependencies from requirements-gpu.txt"
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r (Join-Path $repoRoot "requirements-gpu.txt")
} else {
    Write-Host "Skipping dependency installation"
}

# Keep project root on the module search path so EXAMPLE imports work
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $repoRoot
}

# Verify that TensorFlow sees at least one GPU
Write-Host "Checking GPU visibility (TensorFlow)..."
& $PythonExe -c @"
import sys
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs visible:", ", ".join([g.name for g in gpus]))
    sys.exit(0)
print("No GPU visible to TensorFlow.")
sys.exit(1)
"@
if ($LASTEXITCODE -ne 0) {
    throw "TensorFlow ne détecte pas de GPU. Vérifie les drivers/CUDA/CUDNN avant de relancer."
}

$jnjPath = Join-Path $repoRoot "STATICS/PRICE/JNJ.csv"
$pgPath  = Join-Path $repoRoot "STATICS/PRICE/PG.csv"
if ((-not (Test-Path $jnjPath)) -or (-not (Test-Path $pgPath))) {
    throw "Missing required sample data files: $jnjPath or $pgPath"
}

Write-Host "Running application smoke test (EXAMPLE.RunningScript) with GPU deps ..."
& $PythonExe -m EXAMPLE.RunningScript
