{
  description = "scheduler-gym dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      fenix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        toolchain = fenix.packages.${system}.stable.withComponents [
          "cargo"
          "rustc"
          "rust-src"
          "rustfmt"
          "clippy"
          "rust-analyzer"
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            toolchain
          ]
          ++ (with pkgs; [
            gnumake
            cargo-nextest
            nil
            nixfmt
            ruff
            ty
            uv
            # benchmarking: taskset (bench.py pins case studies), monitoring tools
            util-linux
            linuxPackages.turbostat
            lm_sensors
          ]);

          # szalinski's default make target converts inputs with openscad.
          # (Its mesh-diff targets also need an old CGAL; not provided.)
          buildInputs = with pkgs; [
            openscad
          ];
        };
      }
    );
}
