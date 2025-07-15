{
  description = "A very basic flake for macOS";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-25.05-darwin";
  outputs = { self, nixpkgs }: {
    devShell.aarch64-darwin = nixpkgs.legacyPackages.aarch64-darwin.mkShell {
      buildInputs = with nixpkgs.legacyPackages.aarch64-darwin; [
        pixi
      ];

    shellHook = ''
      '';
    };
  };
}