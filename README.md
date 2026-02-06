# Baseball Tracker

Open-source, budget-friendly baseball swing analysis using computer vision.

## 🎯 Project Vision

A cross-platform baseball tracking system that uses computer vision to capture swing mechanics and ball flight, providing actionable feedback to improve hitting performance.

**Our Edge:**
- Budget-conscious (target: <$500 hardware)
- Open-source and community-driven
- High-quality code following SOLID principles
- Platform-agnostic interface (web-based)

**Primary Competition:** Rapsodo (~$3000+)

## 🏗️ Architecture

- **Backend:** .NET 8 (cross-platform: Windows, Linux, Raspberry Pi)
- **CV/ML:** Python 3.11 (OpenCV, YOLO v8, MediaPipe)
- **Frontend:** Blazor Server (works on iPad, Android, laptop)
- **Storage:** SQLite (local), designed for future cloud sync
- **Deployment:** Docker + docker-compose

## 🚀 Quick Start

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Git

### Setup

1. **Clone repository:**
   ```bash
   git clone <repo-url>
   cd baseball-tracker
   ```

2. **Create required directories:**
   ```bash
   mkdir -p data videos models
   ```

3. **Start services:**
   ```bash
   docker-compose up --build
   ```

4. **Access services:**
   - Python CV API: http://localhost:8000/docs
   - .NET API: http://localhost:5000/swagger
   - Web UI: http://localhost:8080

### First Test

```bash
# Health check
curl http://localhost:8000/health

# Test ball tracking (mock data)
curl -X POST http://localhost:8000/track/ball \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/app/videos/test.mp4"}'
```

## 📁 Project Structure

```
baseball-tracker/
├── src/
│   ├── BaseballTracker.Core/           # Domain models, interfaces
│   ├── BaseballTracker.Infrastructure/ # Data access, camera, Python bridge
│   ├── BaseballTracker.Api/            # REST API + SignalR
│   ├── BaseballTracker.Web/            # Blazor UI
│   └── BaseballTracker.Modules/        # Python CV modules
├── tests/                               # Unit and integration tests
├── docker/                              # Dockerfiles
├── scripts/                             # Setup automation
└── docs/                                # Documentation
```

## 🎥 Hardware Setup

**Recommended Budget Setup (~$300):**
- High-speed camera: 120fps USB camera ($150-200)
- Secondary camera: Standard 60fps webcam ($30-50)
- Compute: Raspberry Pi 4 (4GB+) ($75) OR Windows PC
- Mounting: Steel cage setup

## 🏃 Development

### Running Tests

```bash
# .NET tests
docker-compose run api dotnet test

# Python tests
docker-compose run python-cv pytest
```

### Code Changes

- **Python:** Auto-reloads with uvicorn --reload
- **.NET:** Restart with `docker-compose restart api`

### View Logs

```bash
docker-compose logs -f python-cv
docker-compose logs -f api
```

## 📊 Sprint 1 Status (Current)

- [x] Project scaffolding
- [x] Docker setup
- [x] Core domain models
- [x] Python-to-.NET bridge
- [x] Database repository (SQLite)
- [ ] Ball tracking (real CV - Sprint 2)
- [ ] Bat tracking (real CV - Sprint 2)
- [ ] Web UI
- [ ] Camera capture service

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Module Development

The system is designed with a plugin architecture. See [docs/module-development.md](docs/module-development.md) for creating new analysis modules.

## 📝 License

MIT License - see [LICENSE](LICENSE)

## 🔗 Resources

- [Architecture Documentation](docs/architecture.md)
- [Development Guide](docs/development.md)
- [Hardware Setup Guide](docs/hardware-setup.md)

## 📧 Contact

[Add your contact info or discussion forum link]

---

**Built with ❤️ for the baseball community**
