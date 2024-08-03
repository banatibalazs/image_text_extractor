# Image Text Extractor (OCR)

## Overview
`image_text_extractor` is a Python-based application designed to extract text from images. It supports various image formats and provides a graphical user interface (GUI) for easy interaction. The application leverages libraries such as OpenCV, Pillow, and Tesseract for image processing and text extraction.

## Features
- **Image Loading**: Load images from a folder or individual files.
- **Drawing Modes**: Supports freehand and rectangle drawing modes for selecting regions of interest.
- **Text Extraction**: Extract text from selected regions using Tesseract OCR.
- **Text Copying**: Copy extracted text to a text field with options to preserve text structure.
- **File Dialogs**: Open text and CSV files to display their content in the GUI.

## Requirements
- Python 3.x
- Tesseract OCR

## Installation

### Python Packages
To install the required packages, run the following command:

```sh
pip install -r requirements.txt
```

### Tesseract OCR
1. **Windows**:
   - Download the Tesseract installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
   - Run the installer and follow the instructions.
   - Add the Tesseract installation path (e.g., `C:\Program Files\Tesseract-OCR`) to your system's PATH environment variable.

2. **macOS**:
   - Install Tesseract using Homebrew:
     ```sh
     brew install tesseract
     ```

3. **Linux**:
   - Install Tesseract using your package manager. For example, on Ubuntu:
     ```sh
     sudo apt-get install tesseract-ocr
     ```

## Usage
1. **Run the Application**: Execute the `extractor.py` script to start the application.
2. **Load Images**: Use the file dialog to load images from a folder.
3. **Select Drawing Mode**: Choose between freehand and rectangle drawing modes.
4. **Draw on Image**: Draw on the image to select the region of interest.
5. **Extract Text**: Extract text from the selected region and copy it to the text field.
6. **Save or Copy Text**: Save the extracted text or copy it to the clipboard.

## Code Structure
- `extractor.py`: Main script containing the application logic.
- `README.md`: Documentation file.

## License
This project is licensed under the MIT License.
```