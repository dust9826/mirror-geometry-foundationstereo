# =====================================================================
# setup_local.ps1
# FoundationStereo + mirror-geometry 실험을 로컬(Windows, RTX 5070 Ti)에서
# 돌리기 위한 1회용 세팅 스크립트.
#   - Python venv 생성 (.venv)
#   - Blackwell(sm_120) 지원 torch cu128 설치
#   - requirements.txt 설치
#   - NVlabs/FoundationStereo repo clone (repo/)
#   - 사전학습 모델 다운로드 (pretrained_models/)  -- gdown 폴더
#
# 실행:  powershell -ExecutionPolicy Bypass -File .\setup_local.ps1
# 모델만 다시: powershell -ExecutionPolicy Bypass -File .\setup_local.ps1 -ModelOnly
# =====================================================================
param([switch]$ModelOnly)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot
Write-Host "Project root: $ProjectRoot" -ForegroundColor Cyan

$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"

if (-not $ModelOnly) {
    # ---------------------------------------------------------------- venv
    if (-not (Test-Path $VenvDir)) {
        Write-Host "`n[1/4] Creating venv (.venv) ..." -ForegroundColor Green
        py -3.12 -m venv $VenvDir
    } else {
        Write-Host "`n[1/4] venv already exists -> skip" -ForegroundColor Yellow
    }
    & $VenvPy -m pip install --upgrade pip

    # ---------------------------------------------------------------- torch (Blackwell)
    # repo environment.yml 의 torch==2.4.1(cu121)은 RTX 5070 Ti(sm_120) 미지원.
    # cu128 휠(torch 2.7.x)이 Blackwell 을 네이티브 지원한다.
    Write-Host "`n[2/4] Installing torch cu128 (Blackwell) ..." -ForegroundColor Green
    & $VenvPy -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

    # ---------------------------------------------------------------- 나머지 패키지
    Write-Host "`n[3/4] Installing requirements.txt ..." -ForegroundColor Green
    & $VenvPy -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

    # ---------------------------------------------------------------- repo clone
    $RepoDir = Join-Path $ProjectRoot "repo"
    if (-not (Test-Path (Join-Path $RepoDir "scripts\run_demo.py"))) {
        Write-Host "`nCloning NVlabs/FoundationStereo -> repo/ ..." -ForegroundColor Green
        git clone --depth 1 https://github.com/NVlabs/FoundationStereo.git $RepoDir
    } else {
        Write-Host "`nrepo/ already cloned -> skip" -ForegroundColor Yellow
    }
}

# -------------------------------------------------------------------- 모델 다운로드
Write-Host "`n[4/4] Downloading pretrained model (Google Drive) ..." -ForegroundColor Green
$ModelDir = Join-Path $ProjectRoot "pretrained_models"
$haveModel = Get-ChildItem -Path $ModelDir -Recurse -Filter "*.pth" -ErrorAction SilentlyContinue
if ($haveModel) {
    Write-Host "model(.pth) already present -> skip" -ForegroundColor Yellow
} else {
    & $VenvPy -m gdown --folder "https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf" -O $ModelDir
    Write-Host "if download failed (quota), download manually from:" -ForegroundColor Yellow
    Write-Host "  https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf" -ForegroundColor Yellow
    Write-Host "  and put the '23-51-11' folder (model_best_bp2.pth + cfg.yaml) under pretrained_models/" -ForegroundColor Yellow
}

Write-Host "`nDone." -ForegroundColor Cyan
Write-Host "Next:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\python.exe scripts\prepare_data.py --list" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\python.exe scripts\run_pipeline.py --scene cozy_living_room_baseline080" -ForegroundColor Cyan
