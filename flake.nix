{
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
        # nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };
    outputs = { self, nixpkgs, flake-utils }:
        flake-utils.lib.eachDefaultSystem (system: let
            pkgs = import nixpkgs { inherit system; };
            pythonPackages = ps: with ps; [ mypy ];
            python = pkgs.python3.withPackages pythonPackages;
            nativeBuildInputs = [
                pkgs.wasmtime
                pkgs.wabt
                pkgs.gnuplot
                pkgs.hyperfine
                python
                pkgs.ruff
                pkgs.pyright
            ];
            buildInputs = [ pkgs.python3 pkgs.wasmtime ];

            watim = pkgs.stdenv.mkDerivation {
                inherit nativeBuildInputs buildInputs;
                name = "watim";
                src = ./.;
                buildPhase = ''sh ./bootstrap-native.sh'';
                installPhase = ''
                    mkdir -p $out/bin
                    cp ./watim.wasm $out/bin/
                    echo "#!/usr/bin/env sh" > $out/bin/watim
                    echo "${pkgs.wasmtime}/bin/wasmtime --dir=. -- $out/bin/watim.wasm \"\''${@:1}\"" >> $out/bin/watim
                    chmod +x $out/bin/watim
                '';
            };
        in {
            devShells.default = pkgs.mkShell {
                nativeBuildInputs = (builtins.concatLists [ nativeBuildInputs [ watim pkgs.nodePackages.prettier ] ]);
            };
            packages.default = watim;
        });
}

