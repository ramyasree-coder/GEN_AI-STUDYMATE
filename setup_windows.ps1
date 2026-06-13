<#
Windows setup script for GenAI StudyMate.
Creates a Python 3.11 virtualenv, activates it, installs pip dependencies,
and attempts to install `faiss-cpu`. If faiss installation fails, the script
prints guidance to use conda.

Usage (PowerShell):
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
  ./setup_windows.ps1
#>

Write-Host "GenAI StudyMate Windows Setup"

function Abort($msg) {
    Write-Error $msg
    exit 1
}

# Check python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Abort "Python not found in PATH. Install Python 3.11 and retry."
}

$ver = python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>$null
Write-Host "Found Python version: $ver"
if ($ver -notlike "3.11*") {
    Write-Warning "Python 3.11 is recommended. Proceeding with detected version $ver."
}

Write-Host "Creating virtual environment at .venv"
python -m venv .venv || Abort "Failed to create virtual environment"

Write-Host "Setting process ExecutionPolicy to allow activation"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

Write-Host "Activating virtual environment"
. .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip"
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing pip requirements"
pip install -r requirements.txt

Write-Host "Attempting to install faiss-cpu via pip (may fail on Windows)."
try {
    pip install faiss-cpu>=1.7.0,<2.0.0
    Write-Host "faiss-cpu installed via pip."
} catch {
    Write-Warning "pip install faiss-cpu failed. On Windows it's common to install FAISS via conda."
    Write-Host "If you have conda (Anaconda / Miniconda), run the following commands in PowerShell:"
    Write-Host "  conda create -n studymate python=3.11 -y"
    Write-Host "  conda activate studymate"
    Write-Host "  conda install -c conda-forge faiss-cpu -y"
    Write-Host "  pip install -r requirements.txt"
}

Write-Host "Setup complete. Activate the venv with: . .\\.venv\\Scripts\\Activate.ps1"
Write-Host "Run tests: python test_install.py"
