{
  description = "A very basic flake for macOS with optimized Python";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-25.05-darwin";

  outputs = { self, nixpkgs }: {
    devShell.aarch64-darwin = let
      pkgs = nixpkgs.legacyPackages.aarch64-darwin;
      pythonEnv = pkgs.python311.withPackages (ps: with ps; [
        numpy
        pandas
        scipy
        scikit-learn
        # pytorch-bin or pytorch, if needed
      ]);
    in pkgs.mkShell {
      buildInputs = [
        pkgs.pixi
        pythonEnv
      ];

      shellHook = ''
        echo "Python 3.11 with basic libraries and pixi for mojo is ready..."
      '';
    };
  };
}
