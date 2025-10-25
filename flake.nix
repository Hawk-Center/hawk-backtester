{
  description = "Development environment for hawk-backtester";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; # Or your preferred channel
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux"; # Or detect automatically if needed
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "hawk-backtester-dev";

        # Tools needed for building
        packages = with pkgs; [
          # Rust toolchain
          cargo
          rustc

          # C Compiler (Linker)
          gcc 

          # C++ Standard Library (Runtime for pyarrow etc.)
          stdenv.cc.cc.lib

          # Python Environment Management
          poetry
          python311
          maturin
          # Often needed for Rust crates interacting with C libs
          pkg-config 
          openssl 
        ];

        # Set environment variables
        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
          ]}:$LD_LIBRARY_PATH"
        '';
        # RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
      };
    };
} 