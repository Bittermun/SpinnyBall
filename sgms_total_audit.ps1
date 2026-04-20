# Project Total Audit & Repro Script
Write-Host "--- Sovereign Audit Initiative ---" -ForegroundColor Cyan

# 1. Scaling Audit
Write-Host "[1/3] Executing LOB Scaling & Survivability Audit..." -ForegroundColor Yellow
python lob_scaling.py --audit
if ($LASTEXITCODE -ne 0) { Write-Error "LOB Scaling Audit Failed"; exit 1 }

# 2. Experiment Suite (Repro Mode)
Write-Host "[2/3] Executing Experiment Suite (Repro Mode)..." -ForegroundColor Yellow
python sgms_anchor_suite.py --repro
if ($LASTEXITCODE -ne 0) { Write-Error "Experiment Suite Failed"; exit 1 }

# 3. Test Battery
Write-Host "[3/3] Executing Regression Test Battery..." -ForegroundColor Yellow
python -m pytest tests/
if ($LASTEXITCODE -ne 0) { Write-Error "Test Battery Failed"; exit 1 }

Write-Host "--- Audit Complete: Sovereign Integrity Verified ---" -ForegroundColor Green
