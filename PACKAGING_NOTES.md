# Packaging Notes

## Build Backend
- **Backend:** `maturin`
- **Configuration:** PEP 517 compliant via `pyproject.toml`
- **Rust Integration:** `pyo3` bindings

## Source Layout
- **Rust Crate:** Root directory (standard cargo layout with `src/lib.rs`).
- **Python Source:** `python/` directory (mapped via `python-source` in `pyproject.toml`).
- **Module Name:** `hawk_backtester.hawk_backtester`

## Installation Mechanisms
1. **`pip install .` (Source Build):**
   - Pip reads `pyproject.toml`.
   - Identifies `maturin` as the build backend.
   - Maturin invokes `cargo build --release` to compile the Rust extension.
   - A wheel is built locally and installed.

2. **`pip install <package>` (Binary Wheel):**
   - Pip downloads a pre-compiled wheel from PyPI matching the OS/Arch/Python version.
   - No compilation occurs.
   - **Goal:** Maximize this path.

## Optimization Strategy
- **ABI3:** We aim to enable `abi3` (Stable ABI) so one wheel supports multiple Python versions (e.g., cp38-abi3).
- **CI/CD:** GitHub Actions builds wheels for Linux, macOS, and Windows.
- **Source Optimization:** `Cargo.toml` profiles adjusted for smaller binary size and faster compile times.

