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
   python your_backend_server.py
   ```
   *This will start your MLflow-compatible backend on port 8000*

6. **In a New Terminal, Start the Demo:**
   ```bash
   cd demo
   pip install -r requirements.txt
   python run_app.py
   ```

7. **Access the Demo:**
   - Open `http://localhost:7860` in your browser
   - Or use the AI Studio's port forwarding to access externally

### Option 2: Local Development Setup

1. **Prerequisites:**
   - Python 3.8+
   - CUDA-compatible GPU (recommended)
   - Git

2. **Clone Repository:**
   ```bash
   git clone https://github.com/your-username/AI-Blueprints.git
   cd AI-Blueprints/deep-learning/tryon
   ```

3. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start Backend Server:**
   ```bash
   cd notebook
   python your_backend_server.py
   ```

6. **Start Demo (New Terminal):**
   ```bash
   cd demo
   python run_app.py
   ```

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

# Optional API key for authentication
export API_KEY="your-api-key"

# Health check your backend
curl http://localhost:8000/health
```

### Custom Endpoints

To use a different backend server:

```bash
export MLFLOW_ENDPOINT="http://your-server:8000/"
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

## 📱 Usage Guide

1. **Upload Person Image** (Left Panel):
   - Click or drag-and-drop a photo of a person
   - Best results: full body or upper body shots
   - Supported formats: JPG, PNG

2. **Upload Clothing Item** (Middle Panel):
   - Upload an image of the clothing item
   - Select appropriate clothing type (upper/lower/overall)
   - Adjust inference steps for quality vs speed

3. **Generate Try-On** (Middle Panel):
   - Click "Generate Try-On" button
   - Wait for processing (30 seconds - 2 minutes)
   - View result in the right panel

4. **View Results** (Right Panel):
   - See the generated try-on image
   - Results are automatically saved in `demo/results/`

## 🛠️ Troubleshooting

### Common Issues

**1. OpenCV Import Error:**
```bash
ImportError: libGL.so.1: cannot open shared object file
```
**Solution:** Install system dependencies:
```bash
sudo apt-get update && sudo apt-get install -y libgl1
```

**2. Backend Server Connection Error:**
```bash
Connection error. Please check if the endpoint is available
```
**Solution:** Ensure backend server is running:
```bash
python your_backend_server.py
```

**3. GPU Memory Error:**
```bash
CUDA out of memory
```
**Solution:** Reduce inference steps or image resolution in the demo settings.

**4. Backend Models Not Loaded:**
```bash
Models not loaded yet
```
**Solution:** Wait for backend server to finish loading models or restart server:
```bash
python your_backend_server.py
```

### Performance Tips

- **Faster Results**: Use 10-20 inference steps for quick previews
- **Better Quality**: Use 30-50 inference steps for final results
- **Memory Optimization**: Use smaller image sizes (512x768 instead of 768x1024)
- **Batch Processing**: Process multiple images by restarting the demo

## 🔧 Development

### Backend API Format

Your backend should accept MLflow-compatible requests:

```json
{
  "dataframe_split": {
    "columns": [
      "person_image", "cloth_image", "cloth_type",
      "num_inference_steps", "guidance_scale", "seed",
      "width", "height"
    ],
    "data": [[
      "base64_person_image", "base64_cloth_image", "upper",
      20, 7.5, -1, 512, 768
    ]]
  }
}
```

### Expected Response Format

```json
{
  "predictions": [{
    "result_image": "base64_encoded_result"
  }]
}
```

### Demo API Endpoints

- `GET /` - Main web interface
- `POST /api/tryon` - Virtual try-on processing
- `GET /uploads/<filename>` - Serve uploaded files
- `GET /results/<filename>` - Serve result files

### Adding New Features

1. **Backend Changes**: Modify your backend server
2. **Frontend Changes**: Edit `demo/templates/index.html`
3. **Demo Logic**: Update `demo/app.py`

## 📊 System Requirements

### Minimum Requirements
- **RAM**: 8GB
- **GPU**: 4GB VRAM (GTX 1660 or better)
- **Storage**: 10GB free space
- **Python**: 3.8+

### Recommended Requirements
- **RAM**: 16GB+
- **GPU**: 8GB+ VRAM (RTX 3070 or better)
- **Storage**: 20GB+ free space
- **Python**: 3.10+

## 🆘 Support

For issues and questions:

1. **Check logs**: Look at backend server logs and demo console output
2. **Verify setup**: Ensure all installation steps were completed
3. **Test endpoint**: Verify backend server is responding at `http://localhost:8000/`
4. **Resource check**: Monitor GPU memory and system resources

### Health Check

Test your backend server:
```bash
curl http://localhost:8000/health
```

## 📄 License

This project is part of the AI-Blueprints repository. Please refer to the main project license.

---

🎭 **Ready to try on some virtual outfits?** Start the demo and explore! ✨ 