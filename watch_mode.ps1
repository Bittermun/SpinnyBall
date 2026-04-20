# Watch Mode Script for SpinnyBall Development
# Runs tests automatically on file changes

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SpinnyBall Watch Mode" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Running tests automatically on file changes..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Run pytest-watch with coverage using python
python -m ptw tests/ -- -v --cov=. --cov-report=term-missing
