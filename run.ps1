# PowerShell run script for ECE 49595CV Assignment 2
Write-Host "=== ECE 49595CV Image Classifier ===" -ForegroundColor Cyan
Write-Host "ResNet-improved AlexNet Architecture" -ForegroundColor Cyan
Write-Host "Expected Accuracy: 60-75% on balanced 4-class dataset" -ForegroundColor Yellow
Write-Host ""

# Check if Python is available
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
} else {
    Write-Host "Error: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.7+ and PyTorch" -ForegroundColor Red
    exit 1
}

Write-Host "Using Python: $pythonCmd" -ForegroundColor Green

# Check dependencies
Write-Host "Checking dependencies..." -ForegroundColor Yellow
& $pythonCmd -c "import torch, torchvision, PIL, numpy; print('✓ All dependencies available')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    & $pythonCmd -m pip install -r requirements.txt
}

Write-Host ""

# Run the classifier
Write-Host "Running ResNet-improved CNN classifier..." -ForegroundColor Green
& $pythonCmd main.py

Write-Host ""
Write-Host "Training completed!" -ForegroundColor Cyan
