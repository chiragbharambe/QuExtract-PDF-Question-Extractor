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

def find_text_lines(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (binary.shape[1]//100, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = [cv2.boundingRect(c) for c in contours]
    return sorted(lines, key=lambda r: r[1])

def is_question_start(text, prev_text):
    # Check if there's a newline or empty space before the number
    if not prev_text.strip():
        # Look for number followed by a period or parenthesis
        return bool(re.match(r'^\s*\d+[\.\)]', text.strip()))
    return False

def process_half(half_image, half_binary, output_dir, page_num, half_num):
    lines = find_text_lines(half_binary)
    question_count = 0
    current_question = []
    question_text = ""
    prev_text = ""

    for i, (x, y, w, h) in enumerate(lines):
        line_image = half_image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(line_image, config='--psm 7')

        if is_question_start(text, prev_text) or i == len(lines) - 1:
            if current_question:
                question_count += 1
                y1 = current_question[0][1]
                y2 = current_question[-1][1] + current_question[-1][3]
                question_image = half_image[y1:y2, :]

                # Ensure the image is not too long
                max_height = half_image.shape[0] // 3  # Adjust this value as needed
                if question_image.shape[0] > max_height:
                    question_image = question_image[:max_height]

                filename = f'{output_dir}/page{page_num}_{half_num}_q{question_count}.png'
                cv2.imwrite(filename, question_image)
                print(f"Saved question {question_count} from page {page_num}, half {half_num}")
                print(f"Question text: {question_text.strip()}")

                current_question = []
                question_text = ""

            if i < len(lines) - 1:
                current_question.append((x, y, w, h))
                question_text = text
        else:
            current_question.append((x, y, w, h))
            question_text += " " + text.strip()

        prev_text = text

    return question_count

def process_pdf(pdf_path, output_dir):
    images = convert_from_path(pdf_path, dpi=300)
    total_questions = 0

    for page_num, image in enumerate(images, 1):
        print(f"Processing page {page_num}")
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        binary = preprocess_image(cv_image)

        # Bisect the page vertically
        h, w = cv_image.shape[:2]
        mid = w // 2

        left_half = cv_image[:, :mid]
        right_half = cv_image[:, mid:]
        left_binary = binary[:, :mid]
        right_binary = binary[:, mid:]

        total_questions += process_half(left_half, left_binary, output_dir, page_num, 1)
        total_questions += process_half(right_half, right_binary, output_dir, page_num, 2)

    print(f"Total questions detected: {total_questions}")

# Usage
pdf_path = 'file1.pdf'
output_dir = 'output_questions'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

process_pdf(pdf_path, output_dir)
