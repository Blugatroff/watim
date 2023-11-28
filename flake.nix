{
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };
    outputs = { self, nixpkgs, flake-utils }:
        flake-utils.lib.eachDefaultSystem (system: let
            pkgs = import nixpkgs {
                inherit system;
            };
            pythonPackages = ps: with ps; [
                termcolor
            ];
            python = pkgs.python3.withPackages pythonPackages;
            nativeBuildInputs = [
                pkgs.wasmtime
                pkgs.wabt
                python
                pkgs.gnuplot
                pkgs.hyperfine
            ];
        in {
            devShells.default = pkgs.mkShell {
                inherit nativeBuildInputs;
            };
        });
}

