# Contributing to Baseball Tracker

Thank you for your interest in contributing! This project is open-source and community-driven.

## Ways to Contribute

- 🐛 **Report bugs** - Help us identify issues
- 💡 **Suggest features** - Share your ideas for improvements
- 📝 **Improve documentation** - Help others understand the project
- 🔧 **Write code** - Implement new features or fix bugs
- 🧪 **Add tests** - Improve code quality and coverage
- 🎥 **Share test videos** - Help build our test dataset

## Getting Started

1. **Fork the repository**
2. **Clone your fork**: `git clone https://github.com/YOUR_USERNAME/baseball-tracker.git`
3. **Set up development environment**: `.\scripts\setup.ps1`
4. **Create a branch**: `git checkout -b feature/my-feature`

## Development Process

### Before You Start

- Check existing issues to avoid duplicate work
- For major changes, open an issue first to discuss
- Review [architecture docs](docs/architecture.md)
- Familiarize yourself with [development guide](docs/development.md)

### Writing Code

1. **Follow code style**:
   - C#: Microsoft conventions, nullable reference types
   - Python: PEP 8, type hints
   
2. **Write tests**:
   - Unit tests for new functionality
   - Integration tests for workflows
   - Aim for >80% coverage

3. **Document your code**:
   - XML comments for C# public APIs
   - Docstrings for Python functions
   - Update README if needed

4. **Keep commits atomic**:
   - One logical change per commit
   - Clear commit messages

### Commit Message Format

```
type(scope): brief description

Longer explanation if needed

Fixes #123
```

**Types**: feat, fix, docs, style, refactor, test, chore

**Examples**:
```
feat(cv): add spin rate detection module
fix(api): resolve null reference in swing processing
docs(readme): add hardware setup instructions
test(core): add unit tests for SwingData model
```

### Pull Request Process

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**:
   ```bash
   dotnet test
   docker-compose run python-cv pytest
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/my-feature
   ```

4. **Open Pull Request**:
   - Clear title describing the change
   - Reference related issues
   - Describe what changed and why
   - Include screenshots if UI changes

5. **Address review feedback**:
   - Make requested changes
   - Push updates to same branch
   - Respond to comments

## Code Review Guidelines

### For Contributors

- Be open to feedback
- Explain your approach if needed
- Ask questions if something is unclear

### For Reviewers

- Be constructive and respectful
- Explain the "why" behind suggestions
- Approve when ready, even if minor nitpicks remain

## Areas Needing Help

### High Priority

- [ ] Real ball tracking implementation (OpenCV + YOLO)
- [ ] Real bat tracking implementation
- [ ] Camera calibration system
- [ ] Web UI (Blazor)
- [ ] Integration tests

### Medium Priority

- [ ] Pose estimation module (MediaPipe)
- [ ] Video processing optimization
- [ ] Historical data visualization
- [ ] Export to CSV/Excel

### Nice to Have

- [ ] Mobile app (MAUI)
- [ ] Cloud sync
- [ ] AI-powered feedback
- [ ] Comparison to pro swings
- [ ] Multi-language support

## Adding New CV Modules

We encourage community contributions of new analysis modules!

### Module Template

```python
"""
My Analysis Module
Description of what it does
"""
from api import Point3D
from pydantic import BaseModel

class MyResult(BaseModel):
    metric1: float
    metric2: float
    confidence: float

class MyModule:
    def process_video(self, video_path: str) -> MyResult:
        # Your implementation
        return MyResult(
            metric1=value,
            metric2=value,
            confidence=0.9
        )
```

See [Module Development Guide](docs/module-development.md) for details.

## Testing Your Changes

### Manual Testing

1. **Start services**: `docker-compose up`
2. **Test API**: http://localhost:8000/docs
3. **Check logs**: `docker-compose logs -f`

### Automated Testing

```bash
# .NET tests
dotnet test

# Python tests
docker-compose run python-cv pytest

# With coverage
dotnet test /p:CollectCoverage=true
docker-compose run python-cv pytest --cov=.
```

## Reporting Bugs

### Before Reporting

- Check existing issues
- Verify it's reproducible
- Test on latest code

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
1. Step 1
2. Step 2
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: Windows 11 / macOS / Linux
- Docker version: X.X.X
- Browser (if applicable):

**Logs**
```
Paste relevant logs here
```

**Screenshots**
If applicable
```

## Feature Requests

### Feature Request Template

```markdown
**Problem**
What problem does this solve?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other ways to solve this

**Additional Context**
Mockups, examples, etc.
```

## Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Focus on constructive feedback
- Give credit where due
- Have fun! This is a community project 🎉

## Questions?

- Open a discussion on GitHub
- Check existing documentation
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make Baseball Tracker better! ⚾
