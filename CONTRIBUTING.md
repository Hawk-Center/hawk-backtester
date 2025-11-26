# Contributing to Hawk Backtester

Thank you for your interest in contributing! This guide will help you set up your development environment to build the project from source.

## Development Setup

This project uses **Rust** for the core logic and **Python** for the interface. We use `maturin` to bridge the two.

### Prerequisites
- **Rust**: Stable toolchain (install via [rustup](https://rustup.rs/)).
- **Python**: Version 3.8 or higher.
- **Maturin**: `pip install maturin`

### Setting up a Dev Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Hawk-Center/hawk-backtester.git
   cd hawk-backtester
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install maturin polars
   ```

4. **Build and install in editable mode**:
   ```bash
   maturin develop
   ```
   This compiles the Rust crate and installs it into your current environment. Changes to Python files take effect immediately; changes to Rust files require running `maturin develop` again.

### Using Nix (Optional)
If you use Nix, a flake is provided:
```bash
nix develop
# Configure poetry if needed
poetry env use python3.11
poetry install
poetry run maturin develop
```

## Running Tests

Run Rust tests:
```bash
cargo test
```

Run Python tests (if any):
```bash
pytest
```

## Release Process
We use GitHub Actions to automatically build and publish binary wheels to PyPI on tagged releases.
- Tag a release: `git tag v0.4.0`
- Push the tag: `git push origin v0.4.0`

