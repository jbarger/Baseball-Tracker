# Baseball Tracker - Quick Start Guide

## 🚀 Getting Your Project Running (5 Minutes)

### What You Have

I've generated a complete, working Baseball Tracker project with:
- ✅ Full .NET 8 solution (4 projects)
- ✅ Python FastAPI CV service
- ✅ Docker configuration
- ✅ Database setup (SQLite + EF Core)
- ✅ Stub implementations (returns mock data)
- ✅ Complete documentation
- ✅ Setup scripts

### Your Next Steps

## Step 1: Download & Extract (1 minute)

1. **Download** `baseball-tracker.tar.gz` from this chat
2. **Extract** to your desired location:
   ```bash
   # Windows (PowerShell)
   tar -xzf baseball-tracker.tar.gz
   cd baseball-tracker

   # Or use 7-Zip/WinRAR to extract
   ```

## Step 2: Push to GitHub (2 minutes)

```bash
# Initialize git (if not already done)
cd baseball-tracker
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Sprint 1 scaffolding

- .NET 8 solution with Core, Infrastructure, Api projects
- Python CV modules with FastAPI
- Docker compose setup
- SQLite database with EF Core
- Stub implementations for ball/bat tracking
- Complete documentation and setup scripts"

# Add your GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/baseball-tracker.git

# Push
git branch -M main
git push -u origin main
```

## Step 3: Verify & Run (2 minutes)

```powershell
# Run the setup script
.\scripts\setup.ps1

# This will:
# - Check Docker is installed
# - Create required directories
# - Build Docker images
# - Start all services
# - Run health checks
```

### Expected Output:
```
✅ Docker daemon is running
✅ Docker images built successfully
✅ Services started
✅ Python CV service: http://localhost:8000/docs
✅ .NET API service: http://localhost:5000/swagger
```

## Step 4: Test It Works

Open your browser:

1. **Python CV API**: http://localhost:8000/docs
   - Try POST `/track/ball` with:
     ```json
     {
       "video_path": "/app/videos/test.mp4",
       "camera_id": "default"
     }
     ```
   - You should get back mock data (85-100 mph exit velocity, etc.)

2. **.NET API**: http://localhost:5000/swagger
   - Try GET `/health`
   - Should return: `{"status":"healthy"}`

## What's Working Right Now

✅ **Full project structure** - All directories and files
✅ **Docker services** - API and Python CV running
✅ **Database** - SQLite with EF Core, automatically created
✅ **Python bridge** - .NET can call Python services
✅ **Mock tracking** - Returns realistic fake data
✅ **REST API** - Swagger UI for testing
✅ **Logging** - Full request/response logging

## What's NOT Working Yet (Sprint 2+)

❌ Real computer vision (currently returns mock data)
❌ Camera capture service
❌ Web UI
❌ Real video processing

**This is expected!** Sprint 1 is about scaffolding and infrastructure.

## File Structure You Got

```
baseball-tracker/
├── README.md                    # Project overview
├── LICENSE                      # MIT license
├── CONTRIBUTING.md             # Contribution guide
├── docker-compose.yml          # Docker orchestration
├── .gitignore                  # Git ignore rules
│
├── src/
│   ├── BaseballTracker.Core/          # Domain models
│   │   ├── Models/
│   │   │   ├── SwingData.cs
│   │   │   ├── Session.cs
│   │   │   ├── Player.cs
│   │   │   └── TrackingResults.cs
│   │   └── Interfaces/
│   │       ├── ISwingRepository.cs
│   │       └── IPythonBridge.cs
│   │
│   ├── BaseballTracker.Infrastructure/
│   │   ├── Data/
│   │   │   ├── BaseballTrackerDbContext.cs
│   │   │   ├── SwingRepository.cs
│   │   │   ├── SessionRepository.cs
│   │   │   └── PlayerRepository.cs
│   │   └── Python/
│   │       └── PythonBridge.cs
│   │
│   ├── BaseballTracker.Api/
│   │   ├── Program.cs
│   │   ├── appsettings.json
│   │   └── Controllers/
│   │       └── SwingsController.cs
│   │
│   └── BaseballTracker.Modules/ (Python)
│       ├── api.py
│       ├── requirements.txt
│       ├── ball_tracking/tracker.py
│       └── bat_tracking/tracker.py
│
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.python
│
├── scripts/
│   └── setup.ps1
│
└── docs/
    ├── architecture.md
    └── development.md
```

## Common Issues & Solutions

### "Docker not found"
- Install Docker Desktop: https://www.docker.com/products/docker-desktop
- Restart your terminal after installation

### "Port already in use"
- Stop conflicting services using ports 5000 or 8000
- Or edit `docker-compose.yml` to use different ports

### "Python service not responding"
- Wait 30 seconds (first build takes time)
- Check logs: `docker-compose logs python-cv`
- Rebuild: `docker-compose build python-cv`

### ".NET build errors"
- Check Visual Studio is closed
- Try: `docker-compose down` then `docker-compose up --build`

## Daily Development Workflow

```bash
# Morning - start services
docker-compose up

# During dev - view logs
docker-compose logs -f

# After changes - restart
docker-compose restart api
# (Python auto-reloads)

# Evening - stop services
docker-compose down
```

## What to Do Next

### Option 1: Get Familiar
- Open solution in Visual Studio
- Browse the code
- Read `docs/architecture.md`
- Experiment with Swagger UI

### Option 2: Add Test Videos
- Find baseball swing videos online
- Place in `./videos/test-samples/`
- Test processing (even with mock data)

### Option 3: Start Sprint 2
- Implement real ball tracking (YOLO)
- Implement real bat tracking (OpenCV)
- See GitHub issues for tasks

## Getting Help

- Check `docs/development.md` for detailed guide
- Review `docs/architecture.md` for design decisions
- Open issue on GitHub if stuck
- Share your repo link with me for review!

## Success Checklist

- [ ] Downloaded and extracted project
- [ ] Pushed to GitHub
- [ ] Ran `.\scripts\setup.ps1`
- [ ] Opened http://localhost:8000/docs
- [ ] Tested an endpoint (got mock data back)
- [ ] Opened solution in Visual Studio
- [ ] Read architecture documentation

---

**You're all set!** Share your GitHub repo link and I can review your setup. 🎉⚾
