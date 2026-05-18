# PowerShell script to fix Git complications and push local changes
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Fixing Git Complications and Pushing Changes" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (!(Test-Path ".git")) {
    Write-Host "ERROR: Not in a Git repository directory!" -ForegroundColor Red
    exit 1
}

function RequireGit {
    $g = $(try { git --version 2>&1 } catch { "" })
    if (-not $g) {
        Write-Host "ERROR: git is not available on PATH." -ForegroundColor Red
        exit 1
    }
}

function GitExitOrFail {
    param(
        [Parameter(Mandatory=$true)][string]$CmdLabel
    )
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Git command failed during: $CmdLabel" -ForegroundColor Red
        exit 1
    }
}

RequireGit

# Capture current branch
$currentBranch = $(try { git rev-parse --abbrev-ref HEAD } catch { "unknown" })
Write-Host "`nCurrent branch: $currentBranch" -ForegroundColor Yellow

# Check if .git.bak exists (indicating Git complications)
if (Test-Path ".git.bak") {
    Write-Host "`n[1/6] Git backup detected (.git.bak)! Restoring .git..." -ForegroundColor Red
    Remove-Item -Recurse -Force .git -ErrorAction SilentlyContinue
    Move-Item -Force .git.bak .git
    Write-Host ".git restored from .git.bak" -ForegroundColor Green
} else {
    Write-Host "`n[1/6] No .git.bak found. Proceeding with current repository..." -ForegroundColor Green
}

Write-Host "`n[2/6] Fetching latest changes from remote..." -ForegroundColor Cyan
$null = $(git fetch origin 2>&1)
GitExitOrFail -CmdLabel "git fetch origin"

Write-Host "`n[3/6] Determining best push branch..." -ForegroundColor Cyan

# Determine which branch to push:
# Prefer the current branch *if it exists on origin*; otherwise fallback to origin/main or origin/master.
$remoteHeads = $(git ls-remote --heads origin 2>&1)
$preferBranch = $false

if ($currentBranch -and $currentBranch -ne "HEAD") {
    if ($remoteHeads -match ("refs/heads/" + [regex]::Escape($currentBranch) + "\s*$")) {
        $preferBranch = $true
    }
}

$mainBranch = ""
if ($remoteHeads -match "refs/heads/main\s*$") { $mainBranch = "main" }
elseif ($remoteHeads -match "refs/heads/master\s*$") { $mainBranch = "master" }
else {
    Write-Host "Could not find origin/main or origin/master; defaulting to main." -ForegroundColor Yellow
    $mainBranch = "main"
}

$targetBranch = $mainBranch
if ($preferBranch) {
    $targetBranch = $currentBranch
}

Write-Host "Target branch to push: $targetBranch" -ForegroundColor Yellow

Write-Host "`n[4/6] Switching to target branch..." -ForegroundColor Cyan
# If switching fails (e.g., branch doesn't exist locally), create from origin.
try {
    $null = $(git checkout $targetBranch 2>&1)
    GitExitOrFail -CmdLabel "git checkout $targetBranch"
} catch {
    Write-Host "Local branch $targetBranch not found. Creating from origin/$targetBranch..." -ForegroundColor Yellow
    $null = $(git checkout -b $targetBranch origin/$targetBranch 2>&1)
    GitExitOrFail -CmdLabel "git checkout -b $targetBranch origin/$targetBranch"
}

Write-Host "`n[5/6] Pulling latest changes and preparing to push..." -ForegroundColor Cyan
$null = $(git pull origin $targetBranch --rebase 2>&1)
if ($LASTEXITCODE -ne 0) {
    Write-Host "Rebase pull failed; trying merge pull..." -ForegroundColor Yellow
    $null = $(git pull origin $targetBranch 2>&1)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pull failed; resolve conflicts and re-run." -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n[6/6] Auto-committing local changes (including untracked) if any..." -ForegroundColor Cyan

git status -sb

# Stage all changes (including untracked)
$porcelain = $(git status --porcelain 2>&1)
if ($porcelain -and $porcelain.Trim().Length -gt 0) {
    Write-Host "Changes detected. Staging all files..." -ForegroundColor Yellow
    $null = $(git add -A 2>&1)
    GitExitOrFail -CmdLabel "git add -A"

    # Commit only if there is something to commit
    $staged = $(git diff --cached --name-only 2>&1)
    if ($staged -and $staged.Trim().Length -gt 0) {
        Write-Host "Creating commit..." -ForegroundColor Yellow
        $null = $(git commit -m "Automated commit by fix_and_push.ps1" 2>&1)
        GitExitOrFail -CmdLabel "git commit"
    } else {
        Write-Host "Nothing staged to commit (working tree changed but not staged). Re-check status." -ForegroundColor Yellow
    }
} else {
    Write-Host "No local changes to commit." -ForegroundColor Green
}

Write-Host "`nAttempting to push to origin/$targetBranch..." -ForegroundColor Cyan
$pushOut = $(git push origin $targetBranch 2>&1)
if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully pushed to origin/$targetBranch!" -ForegroundColor Green
} else {
    Write-Host "Push failed:" -ForegroundColor Red
    Write-Host $pushOut -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Successfully completed Git operations!" -ForegroundColor Green
Write-Host "  Pushed updates to $targetBranch" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

