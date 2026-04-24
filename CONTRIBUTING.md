# Contributing to baxter-ts

## Development setup

```bash
git clone https://github.com/PoojaPatil2509/baxter-ts.git
cd baxter-ts
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest tests/test_baxter.py -v           # 37 unit tests
pytest tests/test_comprehensive.py -v   # 30 scenario tests
pytest tests/ -v                         # all 67 tests
pytest tests/ -v --cov=baxter_ts --cov-report=term-missing
```

## Pre-release check

```bash
python pre_launch_check.py
```

All checks must show PASS before any release.

## Release process

```bash
# 1. Bump version in pyproject.toml and baxter_ts/__init__.py
# 2. Commit and push
git add pyproject.toml baxter_ts/__init__.py
git commit -m "chore: bump version to X.Y.Z"
git push origin main

# 3. Build
Remove-Item -Recurse -Force dist, build   # Windows
python -m build

# 4. Verify
twine check dist/*

# 5. Test on TestPyPI
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ baxter-ts

# 6. Publish to PyPI
twine upload dist/*
```

## Raising a pull request

- Fork the repo
- Create a branch: `git checkout -b feat/your-feature`
- Make changes, run all tests
- Open a PR against `main`