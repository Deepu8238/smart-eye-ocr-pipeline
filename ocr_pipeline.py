from __future__ import annotations

import argparse
import re
from typing import Any

import cv2
import pytesseract


def preprocess_image(image_path: str) -> Any:
    """Prepare an image for OCR."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )


def clean_text(text: str) -> str:
    """Normalize OCR output while preserving line structure."""
    lines = [re.sub(r"[ 	]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def extract_text(image_path: str) -> str:
    """Run preprocessing and Tesseract OCR."""
    processed = preprocess_image(image_path)
    return clean_text(pytesseract.image_to_string(processed, config="--psm 6"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple OpenCV + Tesseract OCR pipeline")
    parser.add_argument("image", help="Path to the input image")
    args = parser.parse_args()
    try:
        print(extract_text(args.image))
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
