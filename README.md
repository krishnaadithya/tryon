# AI Virtual Try-On Demo

A simple HTML-based web interface for the CatVTON virtual try-on model using MLflow-compatible backend.

## 🚀 Setup Instructions

### Option 1: AI Studio Setup (Recommended)

1. **Create a New Project in AI Studio:**
   - Go to your AI Studio platform
   - Click "New Project" 
   - **Select "Deep Learning"** as your project type
   - Choose appropriate GPU instance (recommended: T4 or better)

2. **Clone this Repository:**
   ```bash
   git clone https://github.com/your-username/AI-Blueprints.git
   cd AI-Blueprints/deep-learning/tryon
   ```

3. **Install System Dependencies (Required for OpenCV):**
   ```bash
   sudo apt-get update && sudo apt-get install -y libgl1
   ```
   *This is essential for OpenCV to function properly in AI Studio*

4. **Install Python Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the Backend Server:**
   ```bash
   cd notebook
   python register_model.py
   ```
   "This will register model and also test the registered model"

6. **In a New Terminal, Start the Demo:**
   ```bash
   cd demo
   pip install -r requirements.txt
   python run_app.py
   ```

7. **Access the Demo:**
   - Open `http://localhost:7860` in your browser
   - Or use the AI Studio's port forwarding to access externally


## 📁 Project Structure

```
tryon/
├── notebook/
│   ├── your_backend_server.py   # MLflow-compatible backend
│   ├── model/                   # Model components
│   └── utils.py                # Utility functions
├── demo/
│   ├── app.py                  # Flask web application
│   ├── run_app.py              # Application launcher
│   ├── templates/
│   │   └── index.html          # Web interface
│   ├── requirements.txt        # Demo dependencies
│   ├── uploads/               # Uploaded images storage
│   └── results/               # Generated results storage
└── requirements.txt            # Main project dependencies
```

## ⚙️ Configuration

### Environment Variables

```bash
# Backend serving endpoint (default: http://localhost:8000/)
export MLFLOW_ENDPOINT="http://your-server:8000/"
```

### Custom Endpoints

To use a different backend server:

```bash
export MLFLOW_ENDPOINT="ENPOINT URL"
python run_app.py
```

## 🎯 Features

### Web Interface
- **Clean, Modern UI**: Responsive design with drag-and-drop uploads
- **Real-time Preview**: See uploaded images before processing
- **Adjustable Settings**: Control inference steps and clothing type
- **Progress Indicators**: Visual feedback during generation
- **Error Handling**: Clear error messages and troubleshooting

### Model Capabilities
- **Multiple Clothing Types**: Upper body, lower body, full outfits
- **High Quality Results**: Adjustable inference steps (10-50)
- **Fast Processing**: Optimized for real-time inference
- **Memory Efficient**: Works on standard GPU setups

