# Smart Eye OCR Pipeline

A small Python OCR pipeline built for the Smart Eye AI/ML internship technical challenge.

## Approach
1. Load the input image with OpenCV.
2. Convert it to grayscale.
3. Upscale the image to improve character visibility.
4. Apply Gaussian denoising.
5. Apply adaptive thresholding to improve contrast.
6. Run Tesseract OCR and clean the extracted text.

## Requirements
- Python 3.9+
- Tesseract OCR installed and available on PATH

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Install Tesseract:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

## Run
```bash
python ocr_pipeline.py sample/sample_input.png
```

## Sample output
```text
PATHPAL OCR DEMO
Obstacle ahead: bicycle
Distance: 3 meters
```

## Notes
This is a simple baseline designed to be easy to reproduce and extend. For harder images, the next experiments would compare blur, skew and low-light augmentation, stronger preprocessing, and OCR model alternatives using character error rate (CER), word error rate (WER), accuracy and inference time.
