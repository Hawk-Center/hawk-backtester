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

          # Often needed for Rust crates interacting with C libs
          pkg-config 
          openssl 
        ];

        # Optional: Set environment variables if needed
        # RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
      };
    };
} 