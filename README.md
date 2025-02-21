# Face Blurrer

A Python tool that automatically detects and blurs faces in images using MediaPipe face detection. The tool is designed to be robust, handling various lighting conditions and face angles through multiple detection passes.

## Features

- Advanced face detection using multiple MediaPipe models
- Intelligent hair region detection and blurring
- Support for various image formats (PNG, JPG, JPEG, WebP)
- Batch processing of entire directories
- Adaptive blurring with smooth transitions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/blaster151/face-blurrer.git
cd face-blurrer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your images in the `input` directory
2. Run the script:
```bash
python main.py
```
3. Find the processed images in the `output` directory

## Directory Structure

```
face-blurrer/
├── input/          # Place input images here
├── output/         # Blurred images will be saved here
├── main.py         # Main script
└── requirements.txt
```

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- NumPy

## License

MIT License 