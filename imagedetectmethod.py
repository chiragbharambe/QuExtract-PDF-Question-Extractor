import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import re

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # Binarize
    _, binary = cv2.threshold(denoised, 180, 255, cv2.THRESH_BINARY)
    return binary

def detect_text_lines(binary_image):
    # Find contours
    contours, _ = cv2.findContours(255 - binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes for contours
    boxes = [cv2.boundingRect(c) for c in contours]
    
    # Sort boxes by y-coordinate
    boxes.sort(key=lambda b: b[1])
    
    return boxes

def is_question_start(text):
    # Check if the text starts with a number followed by a dot or parenthesis
    return bool(re.match(r'^\d+[\.\)]', text.strip()))

def process_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    
    question_count = 0
    for page_num, image in enumerate(images):
        # Convert PIL Image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess the image
        binary = preprocess_image(cv_image)
        
        # Detect text lines
        lines = detect_text_lines(binary)
        
        current_question = []
        for i, (x, y, w, h) in enumerate(lines):
            line_image = cv_image[y:y+h, x:x+w]
            text = pytesseract.image_to_string(line_image, config='--psm 7')
            
            if is_question_start(text) or (i == len(lines) - 1):
                if current_question:
                    # Save previous question
                    question_count += 1
                    question_image = cv_image[current_question[0][1]:current_question[-1][1]+current_question[-1][3], :]
                    cv2.imwrite(f'question_{question_count}.png', question_image)
                    current_question = []
                
            current_question.append((x, y, w, h))
    
    print(f"Total questions detected: {question_count}")

# Usage
pdf_path = 'file1.pdf'
process_pdf(pdf_path)
