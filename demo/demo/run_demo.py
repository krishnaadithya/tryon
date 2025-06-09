#!/usr/bin/env python3
"""
Simple launcher for the CatVTON Virtual Try-On Demo

Usage:
    python run_demo.py                                    # Use default endpoint
    python run_demo.py --endpoint http://localhost:5000   # Custom endpoint
    python run_demo.py --port 8080                        # Custom port
"""

import argparse
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def parse_args():
    parser = argparse.ArgumentParser(description="Launch CatVTON Virtual Try-On Demo")
    
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:5000/invocations",
        help="MLflow model serving endpoint URL (default: http://localhost:5000/invocations)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key for authentication (optional)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the demo on (default: 7860)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the demo on (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link via Gradio sharing"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set environment variables
    os.environ["MLFLOW_ENDPOINT"] = args.endpoint
    os.environ["API_KEY"] = args.api_key
    
    if args.debug:
        os.environ["GRADIO_DEBUG"] = "1"
    
    print("🚀 CatVTON Virtual Try-On Demo Launcher")
    print("=" * 50)
    print(f"📡 MLflow Endpoint: {args.endpoint}")
    print(f"🔑 API Key: {'Set' if args.api_key else 'Not set'}")
    print(f"🌐 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🔗 Share: {args.share}")
    print(f"🐛 Debug: {args.debug}")
    print("=" * 50)
    
    try:
        from app import VirtualTryOnDemo
        
        demo = VirtualTryOnDemo()
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True,
            inbrowser=not args.debug
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure you have installed the required dependencies:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 