# Contributing to PhishEye

Thank you for your interest in contributing to PhishEye! Together, we can make the web safer against phishing threats and malicious domain infrastructure.

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harshh-0304/PhishEye.git
   cd PhishEye
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies in editable mode:**
   ```bash
   pip install -r PhishEye/requirements.txt
   pip install -e .
   ```

---

## 🧪 Running Tests

Ensure all automated tests pass before submitting a pull request:

```bash
PYTHONPATH=PhishEye python -m unittest discover -s PhishEye/tests -v
```

---

## 📋 Git Commit Guidelines

We adhere to the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` Introduces a new feature or analysis engine
- `fix:` Fixes a bug or erroneous heuristic
- `refactor:` Code restructuring without behavioral change
- `style:` Formatting or CSS visual enhancements
- `test:` Adds or updates automated tests
- `docs:` Documentation improvements
- `ci:` Changes to GitHub Actions or CI/CD pipelines
- `chore:` Dependency, Docker, or tooling maintenance

---

## 🚀 Pull Request Checklist

- [ ] All unit and integration tests pass (`python -m unittest discover -s PhishEye/tests`).
- [ ] No hardcoded absolute file system paths.
- [ ] Documentation and README are updated where applicable.
- [ ] Code conforms to standard Python PEP 8 style conventions.
