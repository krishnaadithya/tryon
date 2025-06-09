# CatVTON Demo Overview

This directory contains a comprehensive demo UI for the CatVTON virtual try-on model with multiple deployment options.

## 📁 File Structure

```
demo/
├── app.py              # Main Gradio demo app (MLflow endpoint)
├── standalone_demo.py  # Standalone demo (direct model loading)
├── run_demo.py        # Launcher script with CLI options
├── setup.py           # Setup and installation script
├── requirements.txt   # Demo-specific dependencies
├── README.md          # Comprehensive documentation
└── OVERVIEW.md        # This file
```

## 🎭 Demo Variants

### 1. **MLflow Endpoint Demo** (`app.py`)
- **Purpose**: Production-ready demo that connects to MLflow serving endpoints
- **Best for**: Deployed models, scalable inference, production environments
- **Requirements**: Running MLflow model server
- **Launch**: `python run_demo.py`
- **Port**: 7860

### 2. **Standalone Demo** (`standalone_demo.py`)
- **Purpose**: Local demo that loads the model directly
- **Best for**: Development, testing, local inference without MLflow
- **Requirements**: Model files and dependencies locally available
- **Launch**: `python standalone_demo.py`
- **Port**: 7861

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   python setup.py
   ```

2. **Choose Demo Type**:
   - **With MLflow**: `python run_demo.py`
   - **Standalone**: `python standalone_demo.py`

3. **Access Interface**:
   - MLflow demo: http://localhost:7860
   - Standalone demo: http://localhost:7861

## ✨ Key Features

### 🎨 Modern UI
- Beautiful gradient styling
- Responsive design
- Intuitive layout
- Real-time progress tracking

### 📸 Flexible Input
- File upload (drag & drop)
- Sample image loading
- Multiple image formats supported
- Image validation and preprocessing

### ⚙️ Advanced Controls
- Inference steps (10-100)
- Guidance scale (1.0-20.0)
- Seed control for reproducibility
- Resolution settings (512-1024px)
- Clothing type selection

### 🎯 Rich Output
- Generated try-on result
- Before/after comparison view
- Mask visualization (optional)
- Download functionality
- Status monitoring

### 🔧 Deployment Options
- MLflow endpoint integration
- Standalone local execution
- Configurable endpoints
- Public sharing via Gradio
- Custom ports and hosts

## 🏗️ Architecture

### MLflow Demo Flow
```
User Input → Gradio UI → HTTP Request → MLflow Endpoint → Model → Response → Display
```

### Standalone Demo Flow
```
User Input → Gradio UI → Direct Model Call → CatVTON Pipeline → Result → Display
```

## 🔌 API Integration

The MLflow demo expects endpoints that match the `VirtualTryOnModel` PyFunc format:

**Request**:
```json
{
  "dataframe_split": {
    "columns": ["person_image", "cloth_image", "cloth_type", ...],
    "data": [["base64_image", "base64_image", "upper", ...]]
  },
  "params": {"return_mask": false}
}
```

**Response**:
```json
{
  "predictions": [{
    "result_image": "base64_result",
    "status": "ok",
    "mask_image": "base64_mask"
  }]
}
```

## 🎯 Usage Scenarios

### 1. **Development & Testing**
- Use `standalone_demo.py` for quick local testing
- Load sample images for rapid iteration
- Adjust parameters and see immediate results

### 2. **Production Deployment**
- Deploy model via MLflow serving
- Use `app.py` as production-ready interface
- Configure load balancing and scaling

### 3. **Research & Experimentation**
- Both demos support parameter tuning
- Mask visualization for analysis
- Reproducible results with seed control

## 🛠️ Configuration

### Environment Variables
```bash
export MLFLOW_ENDPOINT="http://your-endpoint:5000/invocations"
export API_KEY="your-api-key"  # Optional
```

### Command Line Options
```bash
# Custom endpoint
python run_demo.py --endpoint http://server:5000/invocations

# Public sharing
python run_demo.py --share

# Debug mode
python run_demo.py --debug
```

## 📊 Performance Considerations

### GPU vs CPU
- **GPU**: Fast inference (10-30 seconds)
- **CPU**: Slower inference (2-5 minutes)
- Automatic device detection

### Memory Requirements
- **GPU**: 6-8GB VRAM recommended
- **CPU**: 8-16GB RAM recommended
- Adjustable resolution for memory optimization

### Quality vs Speed
- **Fast Preview**: 20-30 steps
- **Balanced**: 50 steps (default)
- **High Quality**: 70-100 steps

## 🔧 Customization

### Styling
- Custom CSS in `app.py` and `standalone_demo.py`
- Gradient themes and modern design
- Responsive layout for different screen sizes

### Functionality
- Easy to add new input types
- Configurable parameter ranges
- Extensible output formats

### Integration
- RESTful API design
- Docker-ready configuration
- Cloud deployment compatible

## 🆘 Troubleshooting

### Common Issues
1. **Model Loading Fails**: Check dependencies and file paths
2. **Connection Errors**: Verify MLflow endpoint is running
3. **Memory Issues**: Reduce image resolution or inference steps
4. **Slow Performance**: Check GPU availability and CUDA setup

### Debug Mode
```bash
python run_demo.py --debug
```

### Logs
- Check console output for detailed error messages
- MLflow server logs for endpoint issues
- Gradio debug mode for UI problems

---

**🎭 Ready to try on some virtual clothes?** Choose your demo variant and start experimenting! ✨ 