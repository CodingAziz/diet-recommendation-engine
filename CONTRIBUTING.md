# Contributing to Diet Recommendation System

Thank you for your interest in contributing to the Diet Recommendation System! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Help create a positive community

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git
- Basic understanding of machine learning and REST APIs

### Quick Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/diet-recommendation-system.git
   cd diet-recommendation-system
   ```
3. **Set up the development environment** (see Development Setup below)
4. **Create a feature branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Option 1: Docker Development (Recommended)

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Run tests
docker-compose -f docker-compose.dev.yml exec api pytest

# View logs
docker-compose -f docker-compose.dev.yml logs -f api
```

### Option 2: Local Development

```bash
# Create virtual environment
cd FastAPI_Backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Using Dev Container (VS Code)

If you're using VS Code, you can use the provided dev container:

1. Install the "Dev Containers" extension
2. Open the project in VS Code
3. When prompted, click "Reopen in Container"
4. The development environment will be automatically set up

## Development Workflow

### 1. Choose an Issue

- Check the [Issues](https://github.com/your-repo/diet-recommendation-system/issues) page
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to indicate you're working on it

### 2. Create a Feature Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-number-description
```

### 3. Make Your Changes

- Write clear, concise commit messages
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes

```bash
# Run the full test suite
pytest tests/ -v --cov=. --cov-report=html

# Run specific tests
pytest tests/test_api.py::TestPredictionEndpoint::test_valid_request -v

# Run performance benchmarks
python -m benchmark

# Check code quality
black --check .
isort --check-only .
mypy .
flake8 .
```

### 5. Update Documentation

- Update API documentation for new endpoints
- Add docstrings to new functions
- Update README.md if needed
- Add examples for new features

## Code Standards

### Python Code Style

This project follows these coding standards:

- **Black**: Code formatting (line length: 127 characters)
- **isort**: Import sorting with black profile
- **mypy**: Static type checking
- **flake8**: Linting with extended ignores for black compatibility

### Code Quality Requirements

- **Test Coverage**: Minimum 90% code coverage
- **Type Hints**: All functions must have type annotations
- **Docstrings**: Use Google-style docstrings for all public functions
- **Error Handling**: Proper exception handling with meaningful messages
- **Logging**: Use structured logging with appropriate log levels

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

Examples:
```
feat(api): add user feedback endpoint

fix(models): handle edge case in hybrid scoring

docs(readme): update installation instructions
```

## Testing

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── test_api.py             # API endpoint tests
├── test_models.py          # ML model tests
├── test_integration.py     # Integration tests
├── test_performance.py     # Performance tests
└── fixtures/
    └── sample_data.json    # Test data fixtures
```

### Writing Tests

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_prediction_endpoint_success():
    """Test successful prediction request"""
    request_data = {
        "nutrition_input": [500, 20, 5, 50, 400, 40, 10, 5, 35],
        "bmi": 24.5,
        "goal": "weight_loss",
        "metric": "nutritional_mae"
    }

    response = client.post("/predict/", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "output" in data
    assert len(data["output"]) > 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run tests matching pattern
pytest -k "test_prediction"

# Run tests in verbose mode
pytest -v

# Run tests and stop on first failure
pytest -x
```

## Submitting Changes

### Pull Request Process

1. **Ensure your branch is up to date**:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run the full test suite** and ensure all checks pass:
   ```bash
   pytest --cov=. --cov-fail-under=90
   black --check .
   isort --check-only .
   mypy .
   ```

3. **Update CHANGELOG.md** if your changes affect users

4. **Create a Pull Request**:
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Link to any related issues
   - Request review from maintainers

5. **Address Review Comments**:
   - Make requested changes
   - Re-run tests to ensure everything still works
   - Update the PR with new commits or force-push if needed

### PR Checklist

- [ ] Tests pass locally and in CI
- [ ] Code follows style guidelines (black, isort, mypy)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated for user-facing changes
- [ ] New dependencies added to requirements.txt
- [ ] Migration scripts included for database changes
- [ ] Security implications reviewed

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Clear title** describing the issue
2. **Steps to reproduce** the problem
3. **Expected behavior** vs actual behavior
4. **Environment details**:
   - OS and version
   - Python version
   - Package versions
   - Browser (for frontend issues)
5. **Error messages** and stack traces
6. **Screenshots** if applicable

### Feature Requests

For feature requests, please include:

1. **Clear description** of the proposed feature
2. **Use case** and why it's needed
3. **Proposed implementation** if you have ideas
4. **Alternatives considered**
5. **Mockups or examples** if applicable

### Security Issues

For security-related issues:
- **DO NOT** create a public GitHub issue
- Email security@your-project.com instead
- Include detailed information about the vulnerability

## Getting Help

- **Documentation**: Check the [API docs](docs/API_DOCUMENTATION.md) and [architecture guide](docs/ARCHITECTURE.md)
- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Search existing issues before creating new ones
- **Slack/Discord**: Join our community chat (if available)

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for their contributions
- GitHub's contributor insights
- Project documentation
- Conference presentations (if applicable)

Thank you for contributing to the Diet Recommendation System! 🎉