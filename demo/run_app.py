#!/usr/bin/env python3
"""
Simple HTML-based Virtual Try-On Demo
Run this script to start the web interface.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import from notebook/
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Set environment variables
os.environ.setdefault("MLFLOW_ENDPOINT", "http://localhost:5000/invocations")

if __name__ == "__main__":
    from app import app
    
    print("🚀 Starting AI Virtual Try-On Demo...")
    print("📱 Open your browser and go to: http://localhost:7860")
    print("🔗 MLflow endpoint:", os.environ.get("MLFLOW_ENDPOINT"))
    print("⚙️  To change the endpoint, set MLFLOW_ENDPOINT environment variable")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=7860) 