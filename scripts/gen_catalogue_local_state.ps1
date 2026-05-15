param(
  [string] = '.',
  [string] = 'catalogue_local_state'
)

Continue = 'Stop'

 = (Resolve-Path ).Path
 = Join-Path  
New-Item -ItemType Directory -Force -Path  | Out-Null

 = Join-Path  'manifest.txt'
if (Test-Path ) { Remove-Item -Force  }

# Skip .git and the output directory itself
 = Get-ChildItem -Force -Recurse -File -Path  |
  Where-Object {
    .FullName -notmatch '\\\\catalogue_local_state\\\\' -and
    .FullName -notmatch '\\\\.git(\\\\|$)'
  }

# Stream write (avoid huge in-memory strings)
 = New-Object System.IO.StreamWriter(, False, [System.Text.Encoding]::UTF8)
try {
  foreach( in ) {
     = .FullName.Substring(.Length + 1)
     = (Get-FileHash -Algorithm SHA256 -Path .FullName).Hash
     = ( + "  \ + .Length + \ \ + )
 .WriteLine()
 }
} finally {
 .Close()
}

Write-Host ('Wrote hashes to ' + + ' (files: ' + .Count + ')')
