# 🎭 CatVTON Virtual Try-On Demo

A modern, beautiful web interface for the CatVTON virtual clothing try-on model, built with Gradio and designed to work with MLflow model serving endpoints.

![Demo Preview](https://via.placeholder.com/800x400/667eea/ffffff?text=CatVTON+Demo+Interface)

## ✨ Features

- **🖼️ Modern UI**: Beautiful, responsive interface with custom styling
- **📱 Multi-input Support**: Upload images, use samples, or capture from webcam
- **⚙️ Advanced Controls**: Fine-tune generation settings and parameters
- **🔄 Real-time Preview**: Before/after comparison and mask visualization
- **📊 Progress Tracking**: Real-time progress updates during generation
- **🔧 Flexible Deployment**: Works with any MLflow serving endpoint
- **🎯 Sample Integration**: Pre-loaded sample images for quick testing

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Access to a running MLflow model serving endpoint
- (Optional) The CatVTON model deployed via the deployment notebook

### Installation

1. **Clone and navigate to the demo directory:**
   ```bash
   cd demo
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the demo:**
   ```bash
   python run_demo.py
   ```

4. **Open your browser** and navigate to `http://localhost:7860`

## 🎮 Usage

### Basic Usage

1. **Upload Images:**
   - Upload a person image (preferably full body or upper body)
   - Upload a clothing item image
   - Or click "Load Sample Images" to test with provided samples

2. **Configure Settings:**
   - Select clothing type: `upper`, `lower`, or `overall`
   - Adjust inference steps (10-100, default: 50)
   - Set guidance scale (1.0-20.0, default: 7.5)
   - Optional: Set a seed for reproducible results

3. **Generate:**
   - Click "✨ Generate Try-On"
   - Watch the progress bar
   - View results in multiple tabs

### Advanced Usage

```bash
# Custom endpoint
python run_demo.py --endpoint http://your-mlflow-server:5000/invocations

# Custom port
python run_demo.py --port 8080

# Public sharing (creates a public URL)
python run_demo.py --share

# With API authentication
python run_demo.py --api-key your-api-key

# Debug mode
python run_demo.py --debug
```

## 🔧 Configuration

### Environment Variables

Set these environment variables to customize the demo:

```bash
export MLFLOW_ENDPOINT="http://localhost:5000/invocations"
export API_KEY="your-api-key"  # Optional
```

### Command Line Options

```
usage: run_demo.py [-h] [--endpoint ENDPOINT] [--api-key API_KEY] [--port PORT] 
                   [--host HOST] [--share] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --endpoint ENDPOINT   MLflow model serving endpoint URL
  --api-key API_KEY     API key for authentication (optional)
  --port PORT           Port to run the demo on (default: 7860)
  --host HOST           Host to run the demo on (default: 0.0.0.0)
  --share               Create a public link via Gradio sharing
  --debug               Enable debug mode
```

## 📡 MLflow Integration

This demo is designed to work with MLflow model serving endpoints. The expected API format matches the `VirtualTryOnModel` PyFunc model from the deployment notebook.

### Expected Request Format

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
      50, 7.5, -1, 768, 1024
    ]]
  },
  "params": {
    "return_mask": false
  }
}
```

### Expected Response Format

```json
{
  "predictions": [{
    "result_image": "base64_encoded_result",
    "status": "ok",
    "mask_image": "base64_encoded_mask"  // if return_mask=true
  }]
}
```

## 🎨 Interface Overview

### Input Section
- **📸 Upload Images**: Direct file upload or drag-and-drop
- **🎯 Sample Images**: Quick-load demonstration images
- **⚙️ Generation Settings**: Control quality and style
- **🔧 Advanced Settings**: Fine-tune resolution and mask display

### Output Section
- **🎨 Try-On Result**: The generated virtual try-on image
- **🔄 Before/After**: Side-by-side comparison view
- **👁️ Mask Visualization**: See the generated mask (optional)

### Status & Progress
- Real-time status updates
- Progress tracking during generation
- Error handling and user feedback

## 🛠️ Troubleshooting

### Common Issues

1. **Connection Error**
   ```
   ❌ Connection error. Please check if the endpoint is available
   ```
   - **Solution**: Verify that your MLflow serving endpoint is running
   - Check the endpoint URL and port
   - Ensure network connectivity

2. **Import Error**
   ```
   ❌ Import error: No module named 'gradio'
   ```
   - **Solution**: Install dependencies: `pip install -r requirements.txt`

3. **Request Timeout**
   ```
   ❌ Request timeout. The model might be processing or unavailable.
   ```
   - **Solution**: The model is taking too long to process
   - Try reducing inference steps or image resolution
   - Check if the model server has sufficient resources

4. **Model Error**
   ```
   ❌ Model error: error message
   ```
   - **Solution**: Check the deployment notebook logs
   - Verify image formats are supported (RGB)
   - Ensure clothing type matches the uploaded garment

### Performance Tips

- **Faster Generation**: Reduce inference steps (20-30 for quick previews)
- **Better Quality**: Increase inference steps (70-100 for final results)
- **Memory Issues**: Reduce image resolution in advanced settings
- **Consistent Results**: Set a fixed seed value

## 🔍 API Testing

You can test the endpoint directly using curl:

```bash
curl -X POST http://localhost:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["person_image", "cloth_image", "cloth_type", "num_inference_steps", "guidance_scale", "seed", "width", "height"],
      "data": [["base64_person_image", "base64_cloth_image", "upper", 50, 7.5, -1, 768, 1024]]
    }
  }'
```

## 📦 Dependencies

Core dependencies:
- `gradio>=4.0.0` - Web interface framework
- `requests>=2.25.0` - HTTP client for API calls
- `pillow>=10.0.0` - Image processing
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This demo interface is part of the CatVTON project. Please refer to the main project license.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review MLflow model serving logs
3. Verify endpoint connectivity
4. Create an issue with detailed error messages

---

**🎭 Happy Virtual Try-On!** ✨ 