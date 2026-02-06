#!/usr/bin/env pwsh
# Baseball Tracker - Master Setup Script
# Run this after cloning the repository

Write-Host @"

██████╗  █████╗ ███████╗███████╗██████╗  █████╗ ██╗     ██╗     
██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗██║     ██║     
██████╔╝███████║███████╗█████╗  ██████╔╝███████║██║     ██║     
██╔══██╗██╔══██║╚════██║██╔══╝  ██╔══██╗██╔══██║██║     ██║     
██████╔╝██║  ██║███████║███████╗██████╔╝██║  ██║███████╗███████╗
╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝
                                                                  
████████╗██████╗  █████╗  ██████╗██╗  ██╗███████╗██████╗         
╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔════╝██╔══██╗        
   ██║   ██████╔╝███████║██║     █████╔╝ █████╗  ██████╔╝        
   ██║   ██╔══██╗██╔══██║██║     ██╔═██╗ ██╔══╝  ██╔══██╗        
   ██║   ██║  ██║██║  ██║╚██████╗██║  ██╗███████╗██║  ██║        
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝        
                                                                  
                    SETUP & VERIFICATION
                                                                  
"@ -ForegroundColor Cyan

Write-Host "This script will verify your setup and start the services.`n" -ForegroundColor White

# ============================================================================
# Step 1: Verify Prerequisites
# ============================================================================

Write-Host "Step 1/4: Checking prerequisites..." -ForegroundColor Yellow

# Check Docker
try {
    $dockerVersion = docker --version
    Write-Host "  ✅ Docker: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Docker not found!" -ForegroundColor Red
    Write-Host "     Please install Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Check Docker Compose
try {
    $composeVersion = docker-compose --version
    Write-Host "  ✅ Docker Compose: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Docker Compose not found!" -ForegroundColor Red
    Write-Host "     Docker Compose should come with Docker Desktop" -ForegroundColor Yellow
    exit 1
}

# Check if Docker is running
try {
    docker ps | Out-Null
    Write-Host "  ✅ Docker daemon is running" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Docker daemon is not running!" -ForegroundColor Red
    Write-Host "     Please start Docker Desktop" -ForegroundColor Yellow
    exit 1
}

# ============================================================================
# Step 2: Create Required Directories
# ============================================================================

Write-Host "`nStep 2/4: Creating required directories..." -ForegroundColor Yellow

$directories = @("data", "videos/captures", "videos/test-samples", "models")

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "  ✅ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  ✓ Exists: $dir" -ForegroundColor Gray
    }
}

# ============================================================================
# Step 3: Build Docker Images
# ============================================================================

Write-Host "`nStep 3/4: Building Docker images..." -ForegroundColor Yellow
Write-Host "  This may take 5-10 minutes on first run..." -ForegroundColor Gray

try {
    docker-compose build
    Write-Host "  ✅ Docker images built successfully" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Docker build failed!" -ForegroundColor Red
    Write-Host "  Check the error messages above" -ForegroundColor Yellow
    exit 1
}

# ============================================================================
# Step 4: Start Services
# ============================================================================

Write-Host "`nStep 4/4: Starting services..." -ForegroundColor Yellow

try {
    docker-compose up -d
    Write-Host "  ✅ Services started" -ForegroundColor Green
    
    Write-Host "`n  Waiting for services to initialize..." -ForegroundColor Gray
    Start-Sleep -Seconds 10
    
} catch {
    Write-Host "  ❌ Failed to start services!" -ForegroundColor Red
    exit 1
}

# ============================================================================
# Health Checks
# ============================================================================

Write-Host "`nRunning health checks..." -ForegroundColor Yellow

# Check Python service
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 5
    if ($response.status -eq "healthy") {
        Write-Host "  ✅ Python CV service: " -NoNewline -ForegroundColor Green
        Write-Host "http://localhost:8000/docs" -ForegroundColor Cyan
    }
} catch {
    Write-Host "  ⚠️  Python CV service not responding yet (may still be starting)" -ForegroundColor Yellow
}

# Check .NET API
try {
    $response = Invoke-RestMethod -Uri "http://localhost:5000/health" -Method Get -TimeoutSec 5
    if ($response.status -eq "healthy") {
        Write-Host "  ✅ .NET API service: " -NoNewline -ForegroundColor Green
        Write-Host "http://localhost:5000/swagger" -ForegroundColor Cyan
    }
} catch {
    Write-Host "  ⚠️  .NET API service not responding yet (may still be starting)" -ForegroundColor Yellow
}

# ============================================================================
# Success Message
# ============================================================================

Write-Host "`n╔════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║          SETUP COMPLETE! 🎉                    ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════╝" -ForegroundColor Green

Write-Host "`n📊 Access your services:" -ForegroundColor Cyan
Write-Host "  • Python CV API:  " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  • .NET API:       " -NoNewline -ForegroundColor White
Write-Host "http://localhost:5000/swagger" -ForegroundColor Cyan
Write-Host "  • Web UI:         " -NoNewline -ForegroundColor White
Write-Host "(Coming in Sprint 4)" -ForegroundColor Gray

Write-Host "`n🎥 Next steps:" -ForegroundColor Cyan
Write-Host "  1. Place test videos in: ./videos/test-samples/" -ForegroundColor White
Write-Host "  2. Test the API using Swagger UI" -ForegroundColor White
Write-Host "  3. Check logs: " -NoNewline -ForegroundColor White
Write-Host "docker-compose logs -f" -ForegroundColor Gray

Write-Host "`n📚 Useful commands:" -ForegroundColor Cyan
Write-Host "  docker-compose logs -f          " -NoNewline -ForegroundColor Gray
Write-Host "View logs" -ForegroundColor White
Write-Host "  docker-compose restart          " -NoNewline -ForegroundColor Gray
Write-Host "Restart services" -ForegroundColor White
Write-Host "  docker-compose down             " -NoNewline -ForegroundColor Gray
Write-Host "Stop services" -ForegroundColor White
Write-Host "  docker-compose up -d            " -NoNewline -ForegroundColor Gray
Write-Host "Start services" -ForegroundColor White

Write-Host "`n📖 Documentation:" -ForegroundColor Cyan
Write-Host "  • Architecture:   ./docs/architecture.md" -ForegroundColor White
Write-Host "  • Development:    ./docs/development.md" -ForegroundColor White
Write-Host "  • Hardware Setup: ./docs/hardware-setup.md" -ForegroundColor White

Write-Host "`nHappy coding! ⚾`n" -ForegroundColor Green
