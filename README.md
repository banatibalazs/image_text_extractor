# Image text extractor

![Python](https://img.shields.io/badge/python-3.x-green.svg)
![Tkinter](https://img.shields.io/badge/tkinter-8.6-green.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.10.0.84-green.svg)
![Pillow](https://img.shields.io/badge/pillow-10.4.0-blue.svg)
![Pytesseract](https://img.shields.io/badge/pytesseract-0.3.10-blue.svg)
![Numpy](https://img.shields.io/badge/numpy-2.0.1-blue.svg)
![PDF2Image](https://img.shields.io/badge/pdf2image-1.17.0-red.svg)

## Overview
`image_text_extractor` is a Python-based application designed to extract text from images.

- **GUI** for image loading and text extraction
- **Manual selection** of regions of interest
- **Text extraction** from selected regions

**Note:** This project was only tested on Windows.


## Demo Video
[![Watch the video](https://img.youtube.com/vi/YG-vw1EUgAk/maxresdefault.jpg)](https://www.youtube.com/watch?v=YG-vw1EUgAk)

### Demo Images

<p>
<img src="images/demo_images/main_menu.png" alt="Main Menu" style="width: 200p;">
<img src="images/demo_images/in_work_preserve_structure_rectangle_mask.png" alt="Mask Drawing Mode Rect" style="width: 400px;">
<img src="images/demo_images/in_work_preserve_structure_free_mask.png" alt="Mask Drawing Mode Free" style="width: 400px;">
<img src="images/demo_images/in_work_csv_mode_rectangle_mask.png" alt="Mask Drawing Mode csv" style="width: 400px;">
</p>

## Features
- **Image Loading**: Load images from a folder or individual files.
- **Mask Drawing Modes**: Supports freehand and rectangle drawing modes for selecting regions of interest.
- **Text Extraction**: Extract text from selected regions using Tesseract OCR.
- **Text Copying**: Copy extracted text to a text field with options to preserve text structure.
- **File Dialogs**: Open text files to display their content in the GUI.

## Requirements
- Python 3.x
- Tesseract OCR
- Python packages: `opencv-python`, `pillow`, `pytesseract`, `numpy`, `matplotlib`

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

2. **macOS** (not tested):
   - Install Tesseract using Homebrew:
     ```sh
     brew install tesseract
     ```

3. **Linux** (not tested):
   - Install Tesseract using your package manager. For example, on Ubuntu:
     ```sh
     sudo apt-get install tesseract-ocr
     ```

### Language Packs
To install additional language packs for Tesseract:

1. **Windows**:
   - Download the desired language pack (e.g., `fra.traineddata` for French) from the [Tesseract GitHub repository](https://github.com/tesseract-ocr/tessdata).
   - Copy the downloaded `.traineddata` file to the `tessdata` directory of your Tesseract installation (e.g., `C:\Program Files\Tesseract-OCR\tessdata`).

2. **macOS** (not tested):
   - Use Homebrew to install the language pack:
     ```sh
     brew install tesseract-lang
     ```

3. **Linux (Ubuntu)** (not tested):
   - Use your package manager to install the language pack. For example, to install the French language pack:
     ```sh
     sudo apt-get install tesseract-ocr-fra
     ```

## Usage
1. **Run the Application**: Execute the `extractor.py` script to start the application.
2. **Load Images**: Use the file dialog to load images from a folder.
3. **Select Drawing Mode**: Choose between freehand and rectangle drawing modes.
4. **Draw on Image**: Draw on the image to select the region of interest.
5. **Extract Text**: Extract text from the selected region and copy it to the text field.
6. **Save or Copy Text**: Save the extracted text or copy it to the clipboard.


**Note**: Ensure to update the tesseract_cmd path in the code to match your Tesseract installation path
```
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```