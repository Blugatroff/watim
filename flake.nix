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
            buildInputs = [ pkgs.wasmtime ];
        in {
            devShells.default = pkgs.mkShell {
                inherit nativeBuildInputs;
            };
            packages.default = pkgs.stdenv.mkDerivation {
                inherit nativeBuildInputs buildInputs;
                name = "watim";
                src = ./.;
                buildPhase = ''bash ./recompile-compiler.sh'';
                installPhase = ''
                    mkdir -p $out/bin
                    cp ./watim.wasm $out/bin/
                    echo "#!/usr/bin/env sh" > $out/bin/watim
                    echo "${pkgs.wasmtime}/bin/wasmtime --dir=. -- ./watim.wasm \"\''${@:1}\"" >> $out/bin/watim
                    chmod +x $out/bin/watim
                '';
            };
        });
}

