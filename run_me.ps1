Param(
    [string]$EnvName = "rl-pytorch-cuda"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Ensure-Conda {
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        return
    }
    throw "conda introuvable. Ouvrez un shell Anaconda/Miniconda ou ajoutez conda au PATH."
}

function Activate-Env {
    param(
        [string]$name
    )

    conda activate $name
}

function Ensure-Streamlit {
    Write-Host "Verification de Streamlit dans l'environnement '$EnvName'..." -ForegroundColor Yellow
    python - <<'PY'
import importlib
mod = importlib.import_module("streamlit")
print(f"Streamlit {mod.__version__} detecte.")
PY
}

function Launch-Streamlit {
    Write-Host "Lancement de Streamlit..." -ForegroundColor Green
    streamlit run "streamlit_app/app.py"
}

# Main
Ensure-Conda
Activate-Env -name $EnvName
Ensure-Streamlit
Launch-Streamlit
