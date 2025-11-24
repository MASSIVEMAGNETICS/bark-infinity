# Contributing to Bark Infinity

Thank you for your interest in contributing to Bark Infinity! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Submitting Changes](#submitting-changes)
7. [Coding Standards](#coding-standards)
8. [Documentation](#documentation)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

## Getting Started

### Areas Where You Can Help

- 🐛 **Bug Fixes**: Fix issues reported in GitHub Issues
- ✨ **Features**: Implement new features from the roadmap
- 📚 **Documentation**: Improve guides, examples, and API docs
- 🧪 **Testing**: Add tests, improve test coverage
- 🎨 **UI/UX**: Enhance web interfaces
- 🚀 **Performance**: Optimize speed and memory usage
- 🌍 **Translations**: Add support for more languages

### Before You Start

1. Check [GitHub Issues](https://github.com/MASSIVEMAGNETICS/bark-infinity/issues) for existing work
2. Comment on an issue to claim it or discuss your approach
3. For major changes, open an issue first to discuss

## Development Setup

### Prerequisites

- Python 3.8 - 3.12
- Git
- (Optional) Docker for container testing
- (Optional) NVIDIA GPU with CUDA for full testing

### Setup Steps

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/bark-infinity.git
   cd bark-infinity
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in Development Mode**
   ```bash
   # Install with all optional dependencies
   pip install -e .[dev,quantization,build,web]
   ```

4. **Verify Installation**
   ```bash
   python -c "import bark_infinity; print(bark_infinity.__version__)"
   bark-infinity info
   ```

### Project Structure

```
bark-infinity/
├── bark_infinity/          # Main package
│   ├── __init__.py
│   ├── api.py             # High-level API
│   ├── generation.py      # Model generation
│   ├── quantization.py    # Quantization support
│   ├── cli.py             # Command-line interface
│   └── ...
├── bark/                   # Original Bark code
├── scripts/                # Build and release scripts
├── .github/workflows/      # CI/CD workflows
├── tests/                  # Test suite (to be added)
├── docs/                   # Documentation (to be added)
└── ...
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-voice-cloning`
- `fix/memory-leak-in-generation`
- `docs/improve-deployment-guide`
- `refactor/simplify-quantization`

### Commit Messages

Follow conventional commits format:

```
type(scope): brief description

Longer description if needed

- Bullet points for details
- Reference issues: Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding/updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `build`: Build system changes
- `ci`: CI/CD changes

Examples:
```
feat(quantization): add 4-bit quantization support

- Implement 4-bit quantization using bitsandbytes
- Add configuration options
- Update documentation

Fixes #123
```

```
fix(cli): handle missing dependencies gracefully

Previously would crash if torch not installed.
Now shows helpful error message.
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test_quantization.py

# Run with coverage
pytest --cov=bark_infinity
```

### Adding Tests

Create tests in `tests/` directory:

```python
# tests/test_new_feature.py
import pytest
from bark_infinity import new_feature

def test_new_feature_basic():
    result = new_feature("input")
    assert result == "expected"

def test_new_feature_edge_case():
    with pytest.raises(ValueError):
        new_feature(None)
```

### Test Guidelines

- Write tests for new features
- Maintain or improve test coverage
- Test edge cases and error conditions
- Use descriptive test names
- Keep tests fast and focused

## Submitting Changes

### Pull Request Process

1. **Update Your Fork**
   ```bash
   git remote add upstream https://github.com/MASSIVEMAGNETICS/bark-infinity.git
   git fetch upstream
   git merge upstream/main
   ```

2. **Create Branch**
   ```bash
   git checkout -b feature/my-feature
   ```

3. **Make Changes**
   - Write code
   - Add tests
   - Update documentation
   - Follow coding standards

4. **Test Locally**
   ```bash
   # Format code
   black bark_infinity/

   # Run tests
   pytest

   # Check types (if applicable)
   mypy bark_infinity/
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add my feature"
   ```

6. **Push to Fork**
   ```bash
   git push origin feature/my-feature
   ```

7. **Create Pull Request**
   - Go to GitHub
   - Click "New Pull Request"
   - Describe your changes
   - Reference related issues
   - Request review

### Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] No merge conflicts
- [ ] CI/CD checks passing
- [ ] Reviewed your own code first

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

## Coding Standards

### Python Style

Follow PEP 8 with these specifics:

- **Line Length**: 100 characters (configured in Black)
- **Imports**: 
  ```python
  # Standard library
  import os
  import sys
  
  # Third-party
  import torch
  import numpy as np
  
  # Local
  from bark_infinity import config
  ```
- **Type Hints**: Use where helpful
  ```python
  def generate_audio(text: str, voice: Optional[str] = None) -> np.ndarray:
      ...
  ```
- **Docstrings**: Use Google style
  ```python
  def my_function(arg1: str, arg2: int) -> bool:
      """Brief description.
      
      Longer description if needed.
      
      Args:
          arg1: Description of arg1
          arg2: Description of arg2
          
      Returns:
          Description of return value
          
      Raises:
          ValueError: When invalid input
      """
  ```

### Code Formatting

Use Black for formatting:

```bash
# Format all code
black bark_infinity/

# Check without modifying
black --check bark_infinity/
```

### Linting

Use flake8 for linting:

```bash
# Run linter
flake8 bark_infinity/

# With specific rules
flake8 bark_infinity/ --max-line-length=100
```

### Type Checking

Use mypy for type checking:

```bash
# Check types
mypy bark_infinity/
```

## Documentation

### Code Documentation

- Add docstrings to all public functions/classes
- Include examples in docstrings when helpful
- Update README.md for user-facing changes
- Update DEPLOYMENT.md for deployment changes
- Add entries to CHANGELOG.md for releases

### Documentation Files

When updating docs:

- **README.md**: Main project overview
- **DEPLOYMENT.md**: Deployment instructions
- **QUICKSTART.md**: Quick reference
- **MOBILE.md**: Mobile access guide
- **CHANGELOG.md**: Version history
- **CONTRIBUTING.md**: This file

### Writing Style

- Use clear, simple language
- Include code examples
- Add screenshots for UI changes
- Link to related documentation
- Keep it up-to-date

## Development Tips

### Useful Commands

```bash
# Install in editable mode
pip install -e .[dev]

# Run single test
pytest -k test_name -v

# Format and lint
black bark_infinity/ && flake8 bark_infinity/

# Build Docker image
docker build -t bark-test .

# Run in Docker
docker run -it bark-test bash

# Check installed version
python -c "import bark_infinity; print(bark_infinity.__version__)"
```

### Debugging

```python
# Enable debug logging
import logging
from bark_infinity import logger
logger.setLevel(logging.DEBUG)

# Or via environment
import os
os.environ['BARK_LOGLEVEL'] = 'DEBUG'
```

### Testing Quantization

```bash
# Test without GPU
BARK_QUANTIZE_8BIT=True python test_quantization.py

# Test with low-compute mode
python -c "
from bark_infinity import setup_low_compute_mode
config = setup_low_compute_mode()
print('Low-compute mode configured')
"
```

## Getting Help

- **Documentation**: Check existing docs first
- **Issues**: Search GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Code**: Look at existing implementations
- **Tests**: Check test files for examples

## Recognition

Contributors will be:
- Listed in CHANGELOG.md for their contributions
- Credited in release notes
- Appreciated in the community!

Thank you for contributing to Bark Infinity! 🎉

---

## Quick Reference

**Setup**: `pip install -e .[dev]`
**Format**: `black bark_infinity/`
**Test**: `pytest`
**Lint**: `flake8 bark_infinity/`
**Build**: `bash scripts/build_docker.sh`

**Questions?** Open an issue or discussion on GitHub!
