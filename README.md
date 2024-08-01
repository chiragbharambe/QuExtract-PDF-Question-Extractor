# QuExtract-PDF-Question-Extractor

This repository contains a data extraction pipeline designed to process PDF files containing questions and extract individual questions as separate images.

## Project Overview

The goal of this project is to automate the extraction of questions from PDF files, particularly those with complex layouts such as multiple columns or varied formatting. This tool is useful for educators, content creators, or anyone working with large volumes of question-based documents.

## Approach

Our approach involves several key steps:

1. **PDF to Image Conversion**: We convert each page of the PDF to a high-resolution image.

2. **Image Preprocessing**: We apply various image processing techniques (grayscale conversion, denoising, binarization) to prepare the image for text detection.

3. **Page Bisection**: To handle potential multi-column layouts, we vertically bisect each page and process each half separately.

4. **Text Line Detection**: We use morphological operations and contour detection to identify individual lines of text.

5. **Question Identification**: We employ regular expressions and contextual analysis to detect the start of new questions.

6. **Question Extraction**: Once a question is identified, we extract it as an image, including all associated text and any figures or diagrams.

7. **Output Generation**: Each extracted question is saved as a separate image file with a descriptive filename.

## Current Limitations and Future Improvements

While our current implementation is functional, there are several areas for potential improvement:

- **Layout Analysis**: Implement more sophisticated layout analysis to handle varied column structures and page layouts.
- **OCR Accuracy**: Explore ways to improve OCR accuracy, especially for complex mathematical or scientific notation.
- **Question Grouping**: Develop methods to group related sub-questions or multi-part questions.
- **Answer Extraction**: Extend the pipeline to also extract and associate answers with their corresponding questions.
- **Metadata Extraction**: Implement extraction of metadata such as subject, difficulty level, or topic.
- **GUI Development**: Create a user-friendly interface for easier use by non-technical users.
- **Output Formats**: Provide options for different output formats (e.g., JSON, CSV) in addition to images.

## Installation and Usage

### Requirements

- Python 3.7 or higher
- pip (Python package installer)
- Tesseract OCR
- Poppler
- Python libraries:
  - opencv-python-headless
  - numpy
  - pytesseract
  - pdf2image

### Quick Setup

1. Clone the repository and navigate to the project directory.

2. Create and activate a virtual environment (optional but recommended).

3. Install the required Python libraries:
   ```
   pip install opencv-python-headless numpy pytesseract pdf2image
   ```

4. Install Tesseract OCR and Poppler on your system (if not already installed).

5. Place your PDF file in the project directory.

6. Update the `pdf_path` variable in the script with your PDF filename.

7. Run the script:
   ```
   python script.py
   ```
Extracted questions will be saved as image files in the `output_questions` directory.

For detailed installation instructions for Tesseract OCR and Poppler, please refer to their official documentation based on your operating system.

## Contributing

We welcome contributions to improve this extraction pipeline. Please feel free to submit issues, feature requests, or pull requests.

