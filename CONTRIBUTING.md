# Contributing to SpinnyBall

Thank you for your interest in contributing to SpinnyBall. This document explains how to contribute.

## Quick Start

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your changes
4. Make your changes
5. Run tests to ensure everything works
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11 or later
- Poetry for dependency management
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/your-username/SpinnyBall.git
cd SpinnyBall

# Install dependencies
poetry install --extras all

# Activate virtual environment
poetry shell
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_multi_body.py
```

## Development Workflow

### Branch Strategy

- `main` - Production code
- `develop` - Development branch
- `feature/your-feature-name` - New features
- `fix/your-bug-fix` - Bug fixes
- `docs/your-doc-change` - Documentation changes

### Making Changes

1. Create a new branch from `develop`
2. Make your changes
3. Write tests for new functionality
4. Update documentation if needed
5. Run tests locally
6. Commit your changes
7. Push to your fork
8. Create a pull request

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add thermal controller for temperature regulation
fix: resolve division by zero in wobble detection
docs: update installation instructions
test: add unit tests for thermal model
```

## Code Style

### Formatting

We use these tools for code quality:

- Black - Code formatting
- Ruff - Linting
- MyPy - Type checking

Run these before committing:

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy .
```

### Python Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for functions and classes
- Keep functions focused and small
- Use descriptive variable names

## Testing

### Test Requirements

- All tests must pass before submitting a pull request
- New features must include tests
- Aim for 70% code coverage or higher
- Test both normal cases and edge cases

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test names
- Group related tests in classes

Example:

```python
def test_wobble_detection_with_valid_signal():
    signal = np.random.randn(1000)
    result = detector.detect_wobble(signal)
    assert result is not None
```

## Documentation

### When to Update Documentation

- Adding new features
- Changing existing functionality
- Updating installation instructions
- Changing API interfaces

### Documentation Style

- Use clear, simple language
- Include code examples
- Explain the purpose and usage
- Update related sections

## Pull Request Process

### Before Submitting

1. Update documentation
2. Add tests for new code
3. Run all tests locally
4. Run code quality checks
5. Rebase on latest `develop` branch

### Pull Request Checklist

- Code follows project style guidelines
- Tests pass locally
- New tests added for new features
- Documentation updated
- Commit messages are clear
- No merge conflicts

### Pull Request Template

When creating a pull request, include:

- Description of changes
- Related issue number
- Testing performed
- Breaking changes (if any)
- Screenshots (for UI changes)

## Getting Help

- Check existing issues for similar problems
- Read the documentation
- Ask questions in issues with the "question" label
- Join discussions for design questions

## Project Structure

- `dynamics/` - Physics simulation code
- `control_layer/` - Control algorithms
- `tests/` - Test files
- `docs/` - Documentation
- `backend/` - API server
- `monte_carlo/` - Monte Carlo simulation

## Areas Needing Help

Check the issues page for items labeled "help wanted" or "good first issue".

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
