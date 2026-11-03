#!/usr/bin/env python3
"""
FTC GPT - Ollama Setup Script for Windows

Downloads and installs Ollama (if not already installed), pulls the required
models, and starts the Ollama service. Safe to run multiple times.

Usage:
    python setup_ollama.py
    python setup_ollama.py --skip-install  (skip Ollama installer, just pull models)
    python setup_ollama.py --models-only   (same as --skip-install)
"""

import subprocess
import sys
import os
import urllib.request
import time
import shutil

OLLAMA_INSTALLER_URL = "https://ollama.com/download/OllamaSetup.exe"
OLLAMA_INSTALLER_PATH = os.path.join(os.environ.get("TEMP", "."), "OllamaSetup.exe")
REQUIRED_MODELS = ["gemma4:e4b", "nomic-embed-text"]
CHECK_INTERVAL = 2
TIMEOUT_SECONDS = 1800  # 30 minutes max for model pulls

def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")

def print_step(number, text):
    print(f"\n[{number}] {text}")

def is_ollama_installed():
    """Check if ollama CLI is available on PATH."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def download_installer():
    """Download the Ollama Windows installer."""
    print(f"Downloading Ollama installer from {OLLAMA_INSTALLER_URL} ...")
    try:
        urllib.request.urlretrieve(OLLAMA_INSTALLER_URL, OLLAMA_INSTALLER_PATH)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to download installer: {e}")
        return False

def run_installer():
    """Run the Ollama installer silently."""
    print("Installing Ollama (this may take a minute)...")
    print("(A brief installer window may appear)")
    try:
        result = subprocess.run(
            [OLLAMA_INSTALLER_PATH],
            capture_output=False, timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        if result.returncode == 0:
            print("Ollama installed successfully!")
            # Clean up installer
            try:
                os.remove(OLLAMA_INSTALLER_PATH)
            except:
                pass
            return True
        else:
            print(f"Installer returned code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("Installer timed out (5 min). Ollama may still be installing in the background.")
        return True
    except Exception as e:
        print(f"ERROR running installer: {e}")
        return False

def wait_for_ollama_ready():
    """Wait for the Ollama service to become ready."""
    print("Waiting for Ollama service to start...", end="", flush=True)
    start = time.time()
    while time.time() - start < 60:
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            if result.returncode == 0:
                print(" Ready!")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        time.sleep(CHECK_INTERVAL)
    print(" Timed out waiting for Ollama.")
    return False

def pull_model(model_name):
    """Pull a model and print progress."""
    print(f"\nPulling model: {model_name}")
    print("-" * 50)
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=False, timeout=TIMEOUT_SECONDS
        )
        if result.returncode == 0:
            print(f"✓ {model_name} ready!")
            return True
        else:
            print(f"✗ Failed to pull {model_name} (exit code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Timed out pulling {model_name}")
        return False
    except FileNotFoundError:
        print("✗ ollama command not found")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ Pull of {model_name} cancelled by user")
        return False

def check_models():
    """Check which models are already downloaded."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        if result.returncode != 0:
            return []
        installed = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if parts:
                installed.append(parts[0])
        return installed
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

def main():
    models_only = "--skip-install" in sys.argv or "--models-only" in sys.argv

    print_header("FTC GPT - Ollama Setup for Windows")
    print("This script will:")
    print("  • Install Ollama (if not already installed)")
    print(f"  • Pull required models: {', '.join(REQUIRED_MODELS)}")
    print()

    if not models_only:
        # Step 1: Check / Install Ollama
        print_step(1, "Checking for Ollama installation...")
        if is_ollama_installed():
            print("✓ Ollama is already installed!")
            # Refresh PATH for current session
            os.environ["PATH"] += os.pathsep + "C:\\Users\\" + os.environ.get("USERNAME", "") + "\\AppData\\Local\\Programs\\Ollama"
        else:
            print("Ollama not found. Installing...")
            if not download_installer():
                sys.exit(1)
            if not run_installer():
                sys.exit(1)
            if not wait_for_ollama_ready():
                sys.exit(1)

    # Step 2: Pull models
    print_step(2, "Checking for required models...")
    installed_models = check_models()
    print(f"Installed models: {installed_models if installed_models else '(none)'}")

    models_to_pull = []
    for model in REQUIRED_MODELS:
        base_name = model.split(":")[0]
        if base_name not in installed_models:
            models_to_pull.append(model)
        else:
            print(f"✓ {model} already downloaded")

    if models_to_pull:
        print(f"\nPulling {len(models_to_pull)} model(s)...")
        print("(This may take several minutes depending on your connection)\n")
        for i, model in enumerate(models_to_pull, 1):
            success = pull_model(model)
            if not success:
                sys.exit(1)
    else:
        print("All required models are already downloaded!")

    # Step 3: Verify
    print_step(3, "Verifying installation...")
    if not is_ollama_installed():
        print("✗ Ollama is not responding")
        sys.exit(1)
    print("✓ Ollama is running and ready!")

    print_header("Setup Complete!")
    print("You can now run FTC GPT:")
    print("  • If you have the .exe: double-click ftc-gpt.exe")
    print("  • If running from source: python ai.py")
    print()
    print("Ollama is running in the background and will auto-start")
    print("on your next reboot as well.")

if __name__ == "__main__":
    main()
