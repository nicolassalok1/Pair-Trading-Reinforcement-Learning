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
    $code = @'
try:
    import importlib
    mod = importlib.import_module("streamlit")
    print(f"Streamlit {mod.__version__} detecte.")
except ImportError as exc:
    import sys
    sys.exit("Streamlit non installe dans l'environnement actif.")
'@
    $code | conda run -n $EnvName python -
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Streamlit manquant; installation via pip..." -ForegroundColor Yellow
        conda run -n $EnvName pip install streamlit | Write-Output
        $code | conda run -n $EnvName python -
        if ($LASTEXITCODE -ne 0) {
            throw "Streamlit introuvable meme apres installation. Verifiez l'environnement '$EnvName'."
        }
    }
}

function Ensure-Protobuf {
    # Streamlit >=1.50 exige protobuf >=3.20 ; TensorFlow 2.10 prefere <3.20.
    # On force une version 3.20.x qui marche avec les deux dans la pratique.
    Write-Host "Ajustement de protobuf (3.20.x) pour compatibilite Streamlit/TensorFlow..." -ForegroundColor Yellow
    conda run -n $EnvName pip install protobuf==3.20.3 | Write-Output
}

function Launch-Streamlit {
    Write-Host "Lancement de Streamlit..." -ForegroundColor Green
    $env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
    conda run -n $EnvName python -m streamlit run "streamlit_app/app.py"
}

# Main
Ensure-Conda
Activate-Env -name $EnvName
Ensure-Streamlit
Ensure-Protobuf
Launch-Streamlit
