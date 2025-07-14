#!/usr/bin/env python3
"""
Setup script for Production Distributed LLM Engineering Guide repository.
Configures environment, installs dependencies, and initializes Azure resources.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import track
from loguru import logger

console = Console()


def run_command(cmd: List[str], description: str, check: bool = True) -> bool:
    """Run a shell command with logging."""
    logger.info(f"Running: {description}")
    console.print(f"[blue]Running:[/blue] {description}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.debug(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Command failed: {e}")
        return False


def install_dependencies(dev: bool = False):
    """Install Python dependencies."""
    console.print("[green]Installing Python dependencies...[/green]")
    
    # Upgrade pip
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip")
    
    # Install requirements
    requirements_file = "requirements.txt"
    if not run_command([sys.executable, "-m", "pip", "install", "-r", requirements_file], 
                      f"Installing from {requirements_file}"):
        console.print("[red]Failed to install requirements[/red]")
        return False
        
    # Install development dependencies
    if dev:
        dev_packages = [
            "pytest>=7.4.0",
            "black>=23.11.0", 
            "ruff>=0.1.6",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
            "jupyter>=1.0.0",
            "ipython>=8.17.0",
        ]
        
        for package in track(dev_packages, description="Installing dev packages..."):
            run_command([sys.executable, "-m", "pip", "install", package], f"Installing {package}")
            
    console.print("[green]✓ Dependencies installed successfully[/green]")
    return True


def setup_pre_commit():
    """Setup pre-commit hooks."""
    console.print("[green]Setting up pre-commit hooks...[/green]")
    
    if not run_command(["pre-commit", "install"], "Installing pre-commit hooks"):
        return False
        
    console.print("[green]✓ Pre-commit hooks installed[/green]")
    return True


def setup_azure_cli():
    """Setup Azure CLI and login."""
    console.print("[green]Setting up Azure CLI...[/green]")
    
    # Check if Azure CLI is installed
    if not run_command(["az", "--version"], "Checking Azure CLI", check=False):
        console.print("[yellow]Azure CLI not found. Please install it from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli[/yellow]")
        return False
        
    # Login to Azure
    console.print("Please login to Azure...")
    if not run_command(["az", "login"], "Azure login"):
        console.print("[red]Azure login failed[/red]")
        return False
        
    console.print("[green]✓ Azure CLI configured[/green]")
    return True


def setup_azure_ml_workspace(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    location: str = "eastus"
):
    """Create Azure ML workspace."""
    console.print(f"[green]Creating Azure ML workspace: {workspace_name}[/green]")
    
    # Create resource group
    run_command([
        "az", "group", "create",
        "--name", resource_group,
        "--location", location
    ], f"Creating resource group {resource_group}")
    
    # Create Azure ML workspace
    if not run_command([
        "az", "ml", "workspace", "create",
        "--name", workspace_name,
        "--resource-group", resource_group,
        "--location", location
    ], f"Creating workspace {workspace_name}"):
        return False
        
    console.print(f"[green]✓ Azure ML workspace {workspace_name} created[/green]")
    return True


def setup_environment_variables():
    """Setup environment variables."""
    console.print("[green]Setting up environment variables...[/green]")
    
    env_vars = {
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_DISABLED": "true",  # Default disabled, enable as needed
    }
    
    env_file = Path(".env")
    with open(env_file, "w") as f:
        f.write("# Distributed LLM Environment Variables\n")
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
            
    console.print(f"[green]✓ Environment variables written to {env_file}[/green]")
    return True


def validate_installation():
    """Validate the installation."""
    console.print("[green]Validating installation...[/green]")
    
    # Test imports
    test_script = """
import torch
import transformers
import datasets
import azure.ai.ml
print("✓ Core packages imported successfully")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ Transformers version: {transformers.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU count: {torch.cuda.device_count()}")
"""
    
    if run_command([sys.executable, "-c", test_script], "Testing imports"):
        console.print("[green]✓ Installation validated successfully[/green]")
        return True
    else:
        console.print("[red]Installation validation failed[/red]")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Production Distributed LLM Engineering Guide")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--azure", action="store_true", help="Setup Azure resources")
    parser.add_argument("--subscription-id", help="Azure subscription ID")
    parser.add_argument("--resource-group", default="distributed-llm-rg", help="Azure resource group")
    parser.add_argument("--workspace-name", default="distributed-llm-workspace", help="Azure ML workspace name")
    parser.add_argument("--location", default="eastus", help="Azure region")
    
    args = parser.parse_args()
    
    console.print("[bold blue]Production Distributed LLM Engineering Guide Setup[/bold blue]")
    console.print("This script will set up your development environment.\n")
    
    success = True
    
    # Install dependencies
    if not install_dependencies(dev=args.dev):
        success = False
        
    # Setup pre-commit hooks
    if args.dev and not setup_pre_commit():
        success = False
        
    # Setup environment variables
    if not setup_environment_variables():
        success = False
        
    # Setup Azure resources
    if args.azure:
        if not setup_azure_cli():
            success = False
        elif args.subscription_id:
            if not setup_azure_ml_workspace(
                args.subscription_id,
                args.resource_group, 
                args.workspace_name,
                args.location
            ):
                success = False
        else:
            console.print("[yellow]--subscription-id required for Azure setup[/yellow]")
            
    # Validate installation
    if not validate_installation():
        success = False
        
    if success:
        console.print("\n[bold green]🎉 Setup completed successfully![/bold green]")
        console.print("\nNext steps:")
        console.print("1. Review the README.md for usage instructions")
        console.print("2. Run example training: python examples/train_code_model.py")
        console.print("3. Start model serving: python src/serving/model_server.py")
        console.print("4. Run benchmarks: python benchmarks/performance_benchmark.py")
    else:
        console.print("\n[bold red]❌ Setup failed![/bold red]")
        console.print("Please check the errors above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()