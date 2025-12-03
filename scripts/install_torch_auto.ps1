<#
Auto-install torch with appropriate CUDA wheel on Windows (PowerShell)
Place this script at the repo root and run it from there.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$backendDir = Join-Path -Path (Get-Location) -ChildPath "backend"
$venvDir = Join-Path -Path $backendDir -ChildPath "venv"

Write-Host "== VTON: Auto Torch + Backend installer (PowerShell) ==" -ForegroundColor Cyan

# Ensure backend dir exists
if (-not (Test-Path $backendDir)) {
    New-Item -ItemType Directory -Path $backendDir | Out-Null
}

# Detect nvidia-smi
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$cudaVersion = $null
if ($null -ne $nvidia) {
    Write-Host "nvidia-smi found. Querying CUDA version..."
    try {
        $raw = & nvidia-smi --query-gpu=cuda_version --format=csv,noheader
        $cudaVersion = $raw.Trim()
    } catch {
        # fallback parse
        $out = & nvidia-smi
        if ($out -match "CUDA Version:\s*([0-9]+\.[0-9]+)") {
            $cudaVersion = $Matches[1]
        }
    }
}

if ([string]::IsNullOrEmpty($cudaVersion)) {
    Write-Warning "Could not auto-detect CUDA version. Please enter (e.g. 12.1 or 11.8) or type 'cpu'"
    $cudaVersion = Read-Host "Enter CUDA version or 'cpu'"
}

Write-Host "Detected/selected CUDA version: $cudaVersion"

# Decide wheel suffix
if ($cudaVersion -eq "cpu") {
    $suffix = "cpu"
} elseif ($cudaVersion.StartsWith("12")) {
    $suffix = "cu121"
} elseif ($cudaVersion.StartsWith("11.8")) {
    $suffix = "cu118"
} elseif ($cudaVersion.StartsWith("11.7")) {
    $suffix = "cu117"
} else {
    Write-Warning "Unrecognized CUDA version: $cudaVersion"
    $suffix = Read-Host "Enter wheel suffix (cu121/cu118/cu117/cpu)"
}

Write-Host "Selected suffix: $suffix"

# Create venv if missing
if (-not (Test-Path (Join-Path $venvDir "Scripts\\Activate.ps1"))) {
    Write-Host "Creating venv at $venvDir"
    python -m venv $venvDir
}

# Activate venv
$activatePath = Join-Path $venvDir "Scripts\\Activate.ps1"
if (-not (Test-Path $activatePath)) {
    throw "Virtualenv activate script not found at $activatePath"
}

Write-Host "Activating venv..."
& $activatePath

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install torch
if ($suffix -eq "cpu") {
    Write-Host "Installing CPU-only torch..."
    pip install torch torchvision torchaudio
} else {
    Write-Host "Installing torch with suffix $suffix..."
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$suffix"
}

# Install backend requirements
$req = Join-Path $backendDir "requirements.txt"
if (Test-Path $req) {
    Write-Host "Installing backend requirements..."
    pip install -r $req
} else {
    Write-Warning "$req not found, skipping."
}

# Verification
Write-Host "Verifying installation..."
python - <<PY
import torch, sys
print("Python exec:", sys.executable)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("CUDA device name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("Device query failed:", e)
PY

Write-Host "Done. To run backend:"
Write-Host "cd backend"
Write-Host "venv\\Scripts\\activate"
Write-Host "uvicorn main:app --host 0.0.0.0 --port 8502 --workers 1"
