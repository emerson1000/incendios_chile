# Script PowerShell para preparar el proyecto para GitHub
# Ejecuta este script en PowerShell: .\preparar_github.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Preparando proyecto para GitHub" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar si git está instalado
try {
    $gitVersion = git --version
    Write-Host "[OK] Git instalado: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Git no está instalado. Instala Git desde: https://git-scm.com/" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Verificar estructura del proyecto
Write-Host "[1] Verificando estructura del proyecto..." -ForegroundColor Yellow

$requiredFiles = @(
    "dashboard.py",
    "requirements.txt",
    "README.md",
    ".gitignore"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] $file existe" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] $file no encontrado" -ForegroundColor Red
    }
}

Write-Host ""

# Verificar si ya es un repo git
if (Test-Path .git) {
    Write-Host "[2] Repositorio Git ya inicializado" -ForegroundColor Yellow
    Write-Host "  Estado actual:" -ForegroundColor Yellow
    git status --short
} else {
    Write-Host "[2] Inicializando repositorio Git..." -ForegroundColor Yellow
    git init
    Write-Host "  [OK] Repositorio inicializado" -ForegroundColor Green
}

Write-Host ""

# Verificar .gitignore
Write-Host "[3] Verificando .gitignore..." -ForegroundColor Yellow
if (Test-Path .gitignore) {
    $gitignoreContent = Get-Content .gitignore -Raw
    if ($gitignoreContent -match "data/processed") {
        Write-Host "  [OK] .gitignore configurado correctamente" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] .gitignore puede necesitar ajustes" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [ERROR] .gitignore no existe" -ForegroundColor Red
}

Write-Host ""

# Verificar datos
Write-Host "[4] Verificando datos..." -ForegroundColor Yellow
$dataFile = "data/processed/conaf_datos_reales_completo.csv"
if (Test-Path $dataFile) {
    $fileSize = (Get-Item $dataFile).Length / 1MB
    Write-Host "  [OK] Dataset encontrado: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Green
    Write-Host "  [INFO] Si el archivo es muy grande, considera no subirlo a GitHub" -ForegroundColor Yellow
} else {
    Write-Host "  [WARNING] Dataset no encontrado. Necesitarás procesar los datos antes de usar Streamlit Cloud" -ForegroundColor Yellow
}

Write-Host ""

# Resumen de próximos pasos
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PRÓXIMOS PASOS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Crear repositorio en GitHub:" -ForegroundColor White
Write-Host "   https://github.com/new" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Conectar repositorio local:" -ForegroundColor White
Write-Host "   git remote add origin https://github.com/TU_USUARIO/incendios-chile.git" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Agregar y commitear archivos:" -ForegroundColor White
Write-Host "   git add ." -ForegroundColor Gray
Write-Host "   git commit -m 'Initial commit'" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Subir a GitHub:" -ForegroundColor White
Write-Host "   git branch -M main" -ForegroundColor Gray
Write-Host "   git push -u origin main" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Deploy en Streamlit Cloud:" -ForegroundColor White
Write-Host "   https://share.streamlit.io" -ForegroundColor Gray
Write-Host ""
Write-Host "Ver setup_for_streamlit_cloud.md para instrucciones detalladas" -ForegroundColor Yellow
Write-Host ""

