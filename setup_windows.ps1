$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

if (!(Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }

$openFaceDir = '.local\openface\OpenFace_2.2.0_win_x64'
$openFaceExe = Join-Path $openFaceDir 'FeatureExtraction.exe'
if (!(Test-Path -LiteralPath $openFaceExe)) {
    New-Item -ItemType Directory -Path '.local' -Force | Out-Null
    $archive = '.local\openface.zip'
    $uri = 'https://github.com/TadasBaltrusaitis/OpenFace/releases/download/OpenFace_2.2.0/OpenFace_2.2.0_win_x64.zip'
    curl.exe -fL --retry 2 -o $archive $uri
    if ($LASTEXITCODE -ne 0) { throw 'OpenFace download failed.' }
    $expected = '4201e96f3bc561a0e3a7fa50d2e12ce782b8a685746d63062572cd5296478dec'
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) {
        throw 'OpenFace archive checksum mismatch.'
    }
    Expand-Archive -LiteralPath $archive -DestinationPath '.local\openface' -Force
}
if (!(Test-Path -LiteralPath (Join-Path $openFaceDir 'model\main_clnf_general.txt'))) {
    throw 'Bundled OpenFace CLNF model is missing.'
}
Write-Output "OpenFace executable: $openFaceExe"
