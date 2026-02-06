# Baseball Tracker - System Architecture

## Overview

The Baseball Tracker system follows a clean, modular architecture based on SOLID principles, designed for extensibility and maintainability.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Web UI (Any Device)                    │
│              Blazor Server / Progressive Web App         │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/SignalR
┌────────────────────▼────────────────────────────────────┐
│              .NET Core API Service                       │
│  (Orchestration, Business Logic, Data Access)           │
└─┬──────────┬───────────┬──────────────┬────────────────┘
  │          │           │              │
  │ Python   │ Python    │ SQLite       │ File System
  │ Interop  │ Interop   │ Repository   │ (Videos/Cache)
  │          │           │              │
┌─▼──────┐ ┌─▼────────┐ ┌▼──────────┐  │
│ Ball   │ │ Swing    │ │ Data      │  │
│ Track  │ │ Analysis │ │ Models    │  │
│ Engine │ │ Engine   │ │           │  │
│(Python)│ │(Python)  │ │           │  │
└─┬──────┘ └─┬────────┘ └───────────┘  │
  │          │                          │
┌─▼──────────▼──────────────────────────▼─┐
│       Camera Capture Service             │
│     (Multi-camera coordination)          │
└──────────────────────────────────────────┘
```

## Technology Stack

### Backend (.NET 8)
- **Language**: C# 12
- **Framework**: ASP.NET Core 8.0
- **ORM**: Entity Framework Core 8.0
- **Database**: SQLite (local), designed for cloud migration
- **Real-time**: SignalR for live updates

### Computer Vision (Python 3.11)
- **API Framework**: FastAPI
- **CV Library**: OpenCV 4.x
- **Object Detection**: Ultralytics YOLO v8
- **Pose Estimation**: MediaPipe (Phase 2)
- **Inference**: ONNX Runtime (optimized for edge)
- **Math**: NumPy, SciPy

### Frontend
- **Framework**: Blazor Server
- **Styling**: Bootstrap 5
- **Charts**: Chart.js
- **Video**: Video.js
- **Target Devices**: iPad, Android tablets, laptops

### DevOps
- **Containerization**: Docker + docker-compose
- **CI/CD**: GitHub Actions (future)
- **Testing**: xUnit (.NET), pytest (Python)

## Project Structure

```
baseball-tracker/
├── src/
│   ├── BaseballTracker.Core/          # Domain layer
│   │   ├── Models/                    # Entities
│   │   └── Interfaces/                # Service contracts
│   │
│   ├── BaseballTracker.Infrastructure/ # Implementation layer
│   │   ├── Data/                      # EF Core repositories
│   │   ├── Python/                    # Python bridge
│   │   └── Camera/                    # Camera services
│   │
│   ├── BaseballTracker.Api/           # Web API
│   │   ├── Controllers/               # REST endpoints
│   │   ├── Hubs/                      # SignalR hubs
│   │   └── Services/                  # Application services
│   │
│   ├── BaseballTracker.Web/           # Blazor UI
│   │   ├── Pages/                     # Razor pages
│   │   └── Components/                # Reusable components
│   │
│   └── BaseballTracker.Modules/       # Python CV modules
│       ├── ball_tracking/             # Ball detection & tracking
│       ├── bat_tracking/              # Bat detection & tracking
│       └── common/                    # Shared utilities
│
├── tests/                              # Test projects
├── docker/                             # Docker configuration
└── docs/                               # Documentation
```

## Design Principles

### 1. Separation of Concerns
- **Core**: Pure business logic, no dependencies
- **Infrastructure**: Implementation details (DB, external services)
- **API**: HTTP layer, orchestration
- **Web**: Presentation layer

### 2. Dependency Inversion
- All modules depend on abstractions (interfaces)
- Implementations are injected via DI container
- Easy to mock for testing

### 3. Plugin Architecture
Analysis modules implement `ISwingAnalysisModule`:
- Ball tracking module
- Bat tracking module
- Pose estimation module (future)
- Spin rate module (future)

New modules can be added without modifying core system.

### 4. Single Responsibility
Each class/module has one reason to change:
- `BallTracker`: Only ball detection/tracking
- `SwingRepository`: Only data persistence
- `PythonBridge`: Only Python↔.NET communication

## Data Flow

### Capture → Process → Store Flow

```
1. Camera Capture
   ↓ (video files)
2. Camera Service
   ↓ (paths + metadata)
3. Swing Processor
   ├→ Python: Ball Tracking
   │  ↓ (BallTrackingResult)
   └→ Python: Bat Tracking
      ↓ (BatTrackingResult)
   ↓ (aggregate)
4. Swing Repository
   ↓ (SwingData entity)
5. Database (SQLite)
```

### Real-time Updates

```
Swing Processor → SignalR Hub → Web Clients
                  (broadcast)    (live update)
```

## Extension Points

### Adding a New Analysis Module

1. Create Python class implementing standard interface:
```python
class MyModule:
    def process_video(self, video_path: str) -> MyResult:
        # Your logic
        pass
```

2. Add endpoint to FastAPI:
```python
@app.post("/analyze/my-module")
async def my_module(request: TrackingRequest):
    return MyModule().process_video(request.video_path)
```

3. Add C# interface method:
```csharp
Task<MyResult> AnalyzeMyModuleAsync(string videoPath);
```

4. System automatically picks it up!

## Deployment Architecture

### Development (Docker Compose)
- All services on single machine
- Hot reload for development
- SQLite database

### Production (Future)
- API: Azure App Service / AWS ECS
- Python: Separate container (GPU optional)
- Database: PostgreSQL / Cosmos DB
- Files: Azure Blob / S3
- CDN: Video delivery

## Security Considerations

- API authentication (future): JWT tokens
- CORS: Configured for trusted origins
- Input validation: All user inputs validated
- File upload: Size limits, type checking
- SQL injection: Prevented by EF Core parameterization

## Performance Targets

- Video processing: <5 seconds for 5-second clip
- API response: <100ms for data queries
- Real-time updates: <1 second latency
- Database: Handle 10,000+ swings without degradation

## Scalability Path

Phase 1 (MVP):
- Single machine
- 1-2 concurrent users
- Local storage

Phase 2:
- Distributed services
- Load balancing
- Cloud storage
- Multiple users

Phase 3:
- Multi-region deployment
- CDN for video
- Microservices architecture
- Millions of swings

---

For more details, see:
- [Development Guide](development.md)
- [Module Development](module-development.md)
- [Hardware Setup](hardware-setup.md)
