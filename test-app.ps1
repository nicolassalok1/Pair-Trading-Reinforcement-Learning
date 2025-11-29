Param(
    [string]$PythonExe = "python",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $repoRoot

Write-Host "Repository root: $repoRoot"
Write-Host "Using Python executable: $PythonExe"

# Ensure the virtual environment exists
$venvPath = Join-Path $repoRoot "venv"
$activatePath = Join-Path $venvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activatePath)) {
    Write-Host "Creating virtual environment at $venvPath"
    & $PythonExe -m venv $venvPath
}

Write-Host "Activating virtual environment"
. $activatePath

if (-not $SkipInstall) {
    Write-Host "Installing dependencies from requirements.txt"
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r (Join-Path $repoRoot "requirements.txt")
} else {
    Write-Host "Skipping dependency installation"
}

# Keep project root on the module search path so EXAMPLE imports work
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $repoRoot
}

$jnjPath = Join-Path $repoRoot "STATICS/PRICE/JNJ.csv"
$pgPath  = Join-Path $repoRoot "STATICS/PRICE/PG.csv"
if ((-not (Test-Path $jnjPath)) -or (-not (Test-Path $pgPath))) {
    throw "Missing required sample data files: $jnjPath or $pgPath"
}

Write-Host "Running application smoke test (EXAMPLE.RunningScript) ..."
& $PythonExe -m EXAMPLE.RunningScript

