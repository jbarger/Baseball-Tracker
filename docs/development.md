# Development Guide

## Getting Started

### Prerequisites
- Docker Desktop
- Visual Studio 2022 (or VS Code with C# extension)
- Git

### Initial Setup

1. **Clone and setup:**
   ```bash
   git clone <repo-url>
   cd baseball-tracker
   .\scripts\setup.ps1
   ```

2. **Verify services:**
   - Python API: http://localhost:8000/docs
   - .NET API: http://localhost:5000/swagger

## Development Workflow

### Running Services

**Start all services:**
```bash
docker-compose up
```

**Start in background:**
```bash
docker-compose up -d
```

**Stop services:**
```bash
docker-compose down
```

**View logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f python-cv
docker-compose logs -f api
```

### Making Code Changes

#### .NET Code Changes

1. Edit files in Visual Studio
2. Restart API container:
   ```bash
   docker-compose restart api
   ```

Alternatively, for faster development:
- Run API locally (not in Docker)
- Start only Python service in Docker

#### Python Code Changes

Python service runs with `--reload` flag, so changes are picked up automatically. Just save the file!

If changes don't appear:
```bash
docker-compose restart python-cv
```

### Adding NuGet Packages

1. Add to `.csproj` file or use Visual Studio
2. Rebuild Docker image:
   ```bash
   docker-compose build api
   docker-compose up -d
   ```

### Adding Python Packages

1. Add to `src/BaseballTracker.Modules/requirements.txt`
2. Rebuild Docker image:
   ```bash
   docker-compose build python-cv
   docker-compose up -d
   ```

## Testing

### Unit Tests (.NET)

```bash
# Run all tests
dotnet test

# Run specific project
dotnet test tests/BaseballTracker.Core.Tests

# With coverage
dotnet test /p:CollectCoverage=true
```

### Unit Tests (Python)

```bash
# Run all tests
docker-compose run python-cv pytest

# Run with coverage
docker-compose run python-cv pytest --cov=.

# Run specific test
docker-compose run python-cv pytest tests/test_ball_tracking.py
```

### Integration Tests

```bash
# Coming in Sprint 2
dotnet test tests/BaseballTracker.Integration.Tests
```

## Database Management

### View Database

```bash
# Install SQLite browser
# Open: ./data/baseball-tracker.db
```

### Reset Database

```bash
# Delete database file
rm ./data/baseball-tracker.db

# Restart API (will recreate)
docker-compose restart api
```

### Migrations (Future)

```bash
# Add migration
dotnet ef migrations add MigrationName --project src/BaseballTracker.Infrastructure

# Apply migration
dotnet ef database update --project src/BaseballTracker.Infrastructure
```

## Debugging

### Debugging .NET in Visual Studio

1. Stop Docker API: `docker-compose stop api`
2. In VS: Set BaseballTracker.Api as startup project
3. Update appsettings.json:
   ```json
   "PythonService": {
     "Endpoint": "http://localhost:8000"
   }
   ```
4. F5 to debug

### Debugging Python

1. Add breakpoint using `import pdb; pdb.set_trace()`
2. Attach to container:
   ```bash
   docker attach baseball-tracker_python-cv_1
   ```

## Common Issues

### Docker won't start
- Ensure Docker Desktop is running
- Check for port conflicts (5000, 8000)
- Try: `docker-compose down` then `docker-compose up`

### Python service errors
- Check logs: `docker-compose logs python-cv`
- Verify requirements.txt has correct versions
- Rebuild: `docker-compose build python-cv`

### .NET build errors
- Check logs: `docker-compose logs api`
- Verify all project references are correct
- Clean solution and rebuild

### Database locked
- Stop all services: `docker-compose down`
- Remove lock files: `rm ./data/*.db-shm ./data/*.db-wal`
- Restart: `docker-compose up`

## Code Style

### C# Guidelines
- Follow Microsoft C# conventions
- Use nullable reference types
- Async/await for I/O operations
- XML comments for public APIs

### Python Guidelines
- Follow PEP 8
- Type hints for function signatures
- Docstrings for all public functions
- Use f-strings for formatting

## Project Structure

```
src/
├── BaseballTracker.Core/          # Domain models, interfaces (no dependencies)
├── BaseballTracker.Infrastructure/ # Data access, external services
├── BaseballTracker.Api/           # REST API, controllers
├── BaseballTracker.Web/           # Blazor UI (Sprint 4)
└── BaseballTracker.Modules/       # Python CV modules
```

### Adding a New Feature

1. **Define interface in Core**:
   ```csharp
   // src/BaseballTracker.Core/Interfaces/IMyService.cs
   public interface IMyService
   {
       Task<Result> DoSomethingAsync();
   }
   ```

2. **Implement in Infrastructure**:
   ```csharp
   // src/BaseballTracker.Infrastructure/MyService.cs
   public class MyService : IMyService
   {
       // Implementation
   }
   ```

3. **Register in API**:
   ```csharp
   // src/BaseballTracker.Api/Program.cs
   builder.Services.AddScoped<IMyService, MyService>();
   ```

4. **Add controller endpoint**:
   ```csharp
   // src/BaseballTracker.Api/Controllers/MyController.cs
   [ApiController]
   [Route("api/[controller]")]
   public class MyController : ControllerBase
   {
       private readonly IMyService _service;
       // ...
   }
   ```

5. **Write tests**:
   ```csharp
   // tests/BaseballTracker.Core.Tests/MyServiceTests.cs
   public class MyServiceTests
   {
       [Fact]
       public async Task DoSomething_ShouldWork()
       {
           // Arrange, Act, Assert
       }
   }
   ```

## Resources

- [ASP.NET Core Docs](https://docs.microsoft.com/aspnet/core)
- [Entity Framework Core](https://docs.microsoft.com/ef/core)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [OpenCV Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

## Getting Help

- Check existing issues on GitHub
- Review architecture docs: `./docs/architecture.md`
- Ask in discussions (if enabled)
- Submit detailed bug reports with logs
