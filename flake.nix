{
  description = "Hawk Backtester - High-performance portfolio backtesting system";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        
        # Python dependencies from pyproject.toml
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          polars
          numpy
          pyarrow
          pytest
          ipykernel
        ]);

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python environment
            pythonEnv
            poetry
            maturin

            # Rust toolchain
            (rust-bin.stable.latest.default.override {
              extensions = [ "rust-src" "rust-analyzer" ];
            })
            cargo
            rustfmt
            clippy

            # Build dependencies
            pkg-config
            openssl

            # Development tools
            git
            direnv
          ];

          # Environment variables
          shellHook = ''
            # Poetry configuration
            export POETRY_VIRTUALENVS_IN_PROJECT=true
            export POETRY_PYTHON_PATH="${pythonEnv}/bin/python"
            
            # Rust configuration
            export RUST_BACKTRACE=1
            
            # For reproducible builds
            export SOURCE_DATE_EPOCH=$(date +%s)
            
            # If on macOS, set deployment target
            if [[ "$OSTYPE" == "darwin"* ]]; then
              export MACOSX_DEPLOYMENT_TARGET=14.0
            fi

            # Create virtual environment if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "Creating Python virtual environment..."
              poetry install
            fi
          '';

          # Library paths for OpenSSL
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
        };
      }
    );
} 