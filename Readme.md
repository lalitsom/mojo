# Mojo Development Environment with Nix and Pixi

This guide outlines how to use a reproducible development environment for the [Mojo programming language](https://docs.modular.com/mojo/) 🔥.

Mojo is a new programming language that extends Python with systems programming features, aiming to combine the usability of Python with the performance of C. This setup uses [Nix](https://nixos.org/) to provide the core tooling and [Pixi](https://pixi.sh/) to manage the project's specific dependencies, including the Mojo SDK.

---

## How to Use This Environment

Once the initial project setup is complete, follow these steps each time you want to work on the project.

### Step 1: Activate the Environment

Each time you open a new terminal, navigate to your project directory and run the following command. This will start a shell that has the `pixi` command available.

```bash
nix develop
```

### Step 2: Run Mojo Commands

You can now use `pixi run` to execute any command within the project environment. For example, to check the Mojo version:

```bash
pixi run mojo --version
```

You should see output similar to this (the version will change depending on the latest nightly build):

```
Mojo 25.5.0.dev2025071505 (7904c89c)
```

To work interactively, you can enter a shell where all tools are directly on your `PATH`:

```bash
pixi shell
```

run a mojo file, you can do device test:

```bash
pixi run mojo device_test.mojo 
```


