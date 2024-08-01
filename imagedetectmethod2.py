import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import re
import os

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    _, binary = cv2.threshold(denoised, 200, 255, cv2.THRESH_BINARY_INV)
    return binary

def detect_columns(binary):
    h, w = binary.shape
    summed_cols = np.sum(binary, axis=0)
    threshold = h * 255 * 0.05  # Adjust this value if needed
    column_separators = np.where(summed_cols < threshold)[0]
    
    if len(column_separators) > 0:
        mid = len(column_separators) // 2
        return [(0, column_separators[mid]), (column_separators[mid], w)]
    else:
        return [(0, w)]

def find_text_lines(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (binary.shape[1]//2, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = [cv2.boundingRect(c) for c in contours]
    return sorted(lines, key=lambda r: r[1])  # Sort by y-coordinate

def is_question_start(text, prev_text):
    # Check if there's a newline or empty space before the number
    if prev_text and not prev_text.strip():
        # Look for bold number followed by a period
        return bool(re.match(r'^\s*\d+\.', text.strip()))
    return False

def process_column(column_image, column_binary, output_dir, page_num, col_num):
    lines = find_text_lines(column_binary)
    question_count = 0
    current_question = []
    prev_text = ""

    for i, (x, y, w, h) in enumerate(lines):
        line_image = column_image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(line_image, config='--psm 7')

        if is_question_start(text, prev_text) or i == len(lines) - 1:
            if current_question:
                question_count += 1
                y1 = current_question[0][1]
                y2 = current_question[-1][1] + current_question[-1][3]
                question_image = column_image[y1:y2, :]
                cv2.imwrite(f'{output_dir}/page{page_num}_col{col_num}_q{question_count}.png', question_image)
                print(f"Saved question {question_count} from page {page_num}, column {col_num}")
                current_question = []

            if i < len(lines) - 1:
                current_question.append((x, y, w, h))
        else:
            current_question.append((x, y, w, h))

        prev_text = text

    return question_count

def process_pdf(pdf_path, output_dir):
    images = convert_from_path(pdf_path, dpi=300)
    total_questions = 0

    for page_num, image in enumerate(images, 1):
        print(f"Processing page {page_num}")
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        binary = preprocess_image(cv_image)
        columns = detect_columns(binary)

        for col_num, (start, end) in enumerate(columns, 1):
            column_image = cv_image[:, start:end]
            column_binary = binary[:, start:end]
            total_questions += process_column(column_image, column_binary, output_dir, page_num, col_num)

    print(f"Total questions detected: {total_questions}")

# Usage
pdf_path = 'file1.pdf'
output_dir = 'output_questions'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

process_pdf(pdf_path, output_dir)
