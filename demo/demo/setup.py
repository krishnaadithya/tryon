#!/usr/bin/env python3
"""
Setup script for CatVTON Demo

This script helps set up the demo environment and install dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command with description and error handling."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_gpu():
    """Check for GPU availability."""
    print("🔍 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  No GPU detected. Demo will run on CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet. GPU check will be performed after installation.")
        return False

def install_requirements():
    """Install required dependencies."""
    print("📦 Installing demo requirements...")
    
    # Install demo-specific requirements first
    demo_reqs = Path(__file__).parent / "requirements.txt"
    if demo_reqs.exists():
        if not run_command(f"pip install -r {demo_reqs}", "Installing demo requirements"):
            return False
    
    # Install main project requirements
    main_reqs = Path(__file__).parent.parent / "requirements.txt"
    if main_reqs.exists():
        if not run_command(f"pip install -r {main_reqs}", "Installing main project requirements"):
            return False
    
    return True

def setup_mlflow():
    """Set up MLflow for model serving."""
    print("🔄 Setting up MLflow...")
    
    # Install MLflow if not present
    if not run_command("pip install mlflow", "Installing MLflow", check=False):
        print("⚠️  MLflow installation failed, continuing anyway...")
    
    # Create MLflow directory structure
    mlflow_dir = Path.home() / "mlflow"
    mlflow_dir.mkdir(exist_ok=True)
    
    print(f"✅ MLflow directory created at: {mlflow_dir}")
    print("💡 To serve your model, run:")
    print("   mlflow models serve -m 'models:/TRU_ON/1' -p 5000 --env-manager local")
    
    return True

def create_launch_scripts():
    """Create convenient launch scripts."""
    print("📝 Creating launch scripts...")
    
    demo_dir = Path(__file__).parent
    
    # Create batch file for Windows
    if platform.system() == "Windows":
        batch_content = """@echo off
echo Starting CatVTON Demo...
cd /d "%~dp0"
python run_demo.py
pause
"""
        with open(demo_dir / "start_demo.bat", "w") as f:
            f.write(batch_content)
        print("✅ Created start_demo.bat for Windows")
    
    # Create shell script for Unix-like systems
    else:
        shell_content = """#!/bin/bash
echo "Starting CatVTON Demo..."
cd "$(dirname "$0")"
python run_demo.py
"""
        script_path = demo_dir / "start_demo.sh"
        with open(script_path, "w") as f:
            f.write(shell_content)
        script_path.chmod(0o755)  # Make executable
        print("✅ Created start_demo.sh for Unix/Linux/macOS")
    
    return True

def verify_installation():
    """Verify that the installation was successful."""
    print("🔍 Verifying installation...")
    
    try:
        import gradio
        print(f"✅ Gradio {gradio.__version__} installed")
        
        import requests
        print("✅ Requests library available")
        
        from PIL import Image
        print("✅ PIL/Pillow available")
        
        import pandas
        print(f"✅ Pandas {pandas.__version__} available")
        
        import numpy
        print(f"✅ NumPy {numpy.__version__} available")
        
        # Check PyTorch
        import torch
        print(f"✅ PyTorch {torch.__version__} available")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda} available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("🎭 CatVTON Demo Setup Complete!")
    print("="*60)
    print("\n📖 USAGE INSTRUCTIONS:")
    print("\n1. 🚀 Start the demo:")
    if platform.system() == "Windows":
        print("   - Double-click 'start_demo.bat'")
        print("   - Or run: python run_demo.py")
    else:
        print("   - Run: ./start_demo.sh")
        print("   - Or run: python run_demo.py")
    
    print("\n2. 🌐 Access the interface:")
    print("   - Open your browser to: http://localhost:7860")
    
    print("\n3. 🎯 For standalone mode (no MLflow needed):")
    print("   - Run: python standalone_demo.py")
    print("   - Access at: http://localhost:7861")
    
    print("\n4. 🔧 Advanced usage:")
    print("   - Custom endpoint: python run_demo.py --endpoint http://your-server:5000/invocations")
    print("   - Custom port: python run_demo.py --port 8080")
    print("   - Public sharing: python run_demo.py --share")
    
    print("\n📡 MLflow Model Serving:")
    print("   If you have a trained model, serve it with:")
    print("   mlflow models serve -m 'models:/TRU_ON/1' -p 5000 --env-manager local")
    
    print("\n🆘 Need help?")
    print("   - Check README.md for detailed instructions")
    print("   - Review troubleshooting section in the documentation")
    print("\n" + "="*60)

def main():
    """Main setup function."""
    print("🎭 CatVTON Demo Setup")
    print("="*40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check GPU
    check_gpu()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Setup MLflow
    setup_mlflow()
    
    # Create launch scripts
    create_launch_scripts()
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    # Print usage instructions
    print_usage_instructions()

if __name__ == "__main__":
    main() 