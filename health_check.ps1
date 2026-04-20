# Health Check Script for SpinnyBall Project
# Runs all quality checks before commits

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SpinnyBall Health Check" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$errors = 0
$warnings = 0

# Check Python syntax for core modules
Write-Host "[1/5] Checking Python syntax..." -ForegroundColor Cyan
$python_files = @(
    "control_layer/*.py",
    "dynamics/*.py",
    "monte_carlo/*.py",
    "tests/*.py"
)

$syntax_errors = 0
foreach ($pattern in $python_files) {
    $files = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        python -m py_compile $file.FullName
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ✗ Syntax error in $($file.Name)" -ForegroundColor Red
            $syntax_errors++
        }
    }
}

if ($syntax_errors -eq 0) {
    Write-Host "  ✓ No syntax errors found" -ForegroundColor Green
} else {
    Write-Host "  ✗ $syntax_errors files have syntax errors" -ForegroundColor Red
    $errors++
}
Write-Host ""

# Run unit tests
Write-Host "[2/5] Running unit tests..." -ForegroundColor Cyan
python -m pytest tests/ -v -x --tb=short
$test_exit_code = $LASTEXITCODE

if ($test_exit_code -eq 0) {
    Write-Host "  ✓ All tests passed" -ForegroundColor Green
} else {
    Write-Host "  ✗ Some tests failed" -ForegroundColor Red
    $errors++
}
Write-Host ""

# Check test coverage
Write-Host "[3/5] Checking test coverage..." -ForegroundColor Cyan
python -m pytest tests/ --cov=. --cov-report=term-missing --cov-fail-under=70
$coverage_exit_code = $LASTEXITCODE

if ($coverage_exit_code -eq 0) {
    Write-Host "  ✓ Coverage meets 70% threshold" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Coverage below 70% threshold" -ForegroundColor Yellow
    $warnings++
}
Write-Host ""

# Run linting with ruff
Write-Host "[4/5] Running linting..." -ForegroundColor Cyan
ruff check .
$ruff_exit_code = $LASTEXITCODE

if ($ruff_exit_code -eq 0) {
    Write-Host "  ✓ No linting errors" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Linting issues found (non-blocking)" -ForegroundColor Yellow
    $warnings++
}
Write-Host ""

# Check for critical Phase 3 tests
Write-Host "[5/5] Checking Phase 3 critical tests..." -ForegroundColor Cyan
$phase3_tests = @(
    "tests/test_anomaly_detection.py",
    "tests/test_latency_injection.py"
)

$phase3_errors = 0
foreach ($test_file in $phase3_tests) {
    if (Test-Path $test_file) {
        python -m pytest $test_file -v -x
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ✗ Phase 3 test failed: $test_file" -ForegroundColor Red
            $phase3_errors++
        }
    }
}

if ($phase3_errors -eq 0) {
    Write-Host "  ✓ Phase 3 tests passed" -ForegroundColor Green
} else {
    Write-Host "  ✗ $phase3_errors Phase 3 tests failed" -ForegroundColor Red
    $errors++
}
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Health Check Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "✓ ALL CHECKS PASSED" -ForegroundColor Green
    Write-Host "  Project is healthy and ready to commit" -ForegroundColor Green
    exit 0
} elseif ($errors -eq 0 -and $warnings -gt 0) {
    Write-Host "⚠ CHECKS PASSED WITH WARNINGS" -ForegroundColor Yellow
    Write-Host "  $warnings warnings found (non-blocking)" -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "✗ CHECKS FAILED" -ForegroundColor Red
    Write-Host "  $errors errors, $warnings warnings" -ForegroundColor Red
    Write-Host "  Please fix errors before committing" -ForegroundColor Red
    exit 1
}
