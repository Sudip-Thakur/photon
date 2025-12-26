# AI Colorization Studio

A professional desktop application for real-time grayscale to RGB image colorization using Enhanced Pix2Pix models with PyQt5.

## Features

- **Real-time IR Camera Colorization** - Live feed from IR camera via UDP stream
- **Image File Colorization** - Process individual images
- **Batch Processing** - Colorize multiple images at once
- **Model Management** - Load and manage trained models
- **Side-by-Side Comparison** - View original and colorized output together
- **Snapshot & Recording** - Capture frames and record video

## Requirements

- Python 3.8+
- PyQt5
- PyTorch
- OpenCV
- PyAV (av==14.0.0)
- NumPy

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd D:\Hackathon\Python_GUI\colorization_app
   ```

2. **Create and activate virtual environment (recommended):**
   ```bash
   python -m venv colorization_env
   colorization_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install av==14.0.0
   ```

## Usage

### Running the Application

**Basic usage (default IR camera settings):**
```bash
python main.py
```

**With custom IR camera IP and port:**
```bash
python main.py --ip <IR_CAMERA_IP> --port <PORT>
```

### Examples

```bash
# Default IP (192.168.137.1:9000)
python main.py

# Custom IP address with default port
python main.py --ip 192.168.8.20

# Custom IP and port
python main.py --ip 192.168.1.100 --port 8000
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ip` | `192.168.137.1` | IR camera IP address |
| `--port` | `9000` | IR camera UDP port |

## IR Camera Setup

The application uses an IR camera that streams video over UDP. Make sure your IR camera is:

1. **Powered on** and connected to the network
2. **Streaming** to the specified IP and port
3. **Accessible** from your computer (check firewall settings)

### Camera Configuration

The IR camera settings can be configured in two ways:

1. **Command line arguments** (recommended):
   ```bash
   python main.py --ip YOUR_IP --port YOUR_PORT
   ```

2. **Edit default values** in `utils/camera.py`:
   ```python
   IR_CAMERA_IP = "192.168.137.1"
   IR_CAMERA_PORT = 9000
   ```

## Application Controls

### Camera Section
- **Start** - Connect to IR camera and start live feed
- **Stop** - Disconnect from IR camera
- **Snapshot** - Capture current frame

### Image Section
- **Load Image** - Load an image file for colorization
- **Process** - Colorize the loaded image
- **Save Result** - Save the colorized output

### Model Section
- **Load Model** - Load a trained colorization model (.pth file)

## Project Structure

```
colorization_app/
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── model/
│   ├── enhanced_pix2pix.py
│   └── model_loader.py
├── ui/
│   ├── main_window.py
│   ├── comparison_view.py
│   └── settings_dialog.py
└── utils/
    ├── camera.py        # IR camera capture (PyAV)
    ├── image_utils.py
    └── file_handler.py
```

## Troubleshooting

### App freezes when starting camera
- Check if IR camera is streaming
- Verify IP address and port are correct
- Check firewall allows UDP traffic

### "No video stream found" error
- Ensure IR camera is actively streaming
- Verify the UDP port is correct
- Check network connectivity

### "Connection timed out" error
- IR camera may not be reachable
- Check if IP address is correct
- Ensure camera and PC are on same network

### Model loading warnings
- PyTorch security warnings about `weights_only` can be ignored if you trust the model source

## Dependencies

```
PyQt5>=5.15.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
av==14.0.0
Pillow>=10.0.0
tqdm>=4.65.0
```

## License

AI Colorization Studio - All rights reserved.

## Author

AI Colorization Studio Team