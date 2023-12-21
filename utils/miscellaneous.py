import csv 
import uuid
import time
import fitz 
import io
import os
import re
import nltk
import pytz
from PIL import Image 
from pytesseract import pytesseract 
from nltk.corpus import stopwords
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from collections import defaultdict
from img2table.document import PDF
from img2table.ocr import TesseractOCR
from datetime import datetime

def generate_next_id(previous_id):
    prefix = previous_id[:-4]
    number = int(previous_id[-4:]) + 1

    if number > 9999:
        prefix = chr((ord(prefix[0]) - ord('a') + 1) % 26 + ord('a')) + prefix[1:]
        number = 1

    return f"{prefix}{number:04d}"

def get_last_id(csv_filename):
    try:
        with open(csv_filename, 'r') as csv_file:
            reader = csv.reader(csv_file)
            rows = list(reader)
            if len(rows) > 1:
                last_entry_id = rows[-1][0]
                if len(last_entry_id) == 7 and last_entry_id[:3].isalpha() and last_entry_id[3:].isdigit():
                    return last_entry_id
    except FileNotFoundError:
        # If the file doesn't exist, return None
        pass
    return None

def append_to_master_csv(PDF_PATH):
    """
    Append data to a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.
    - data (list): List of data to be appended to the CSV file.
    """
    csv_filename = 'master.csv'  
    last_id = get_last_id(csv_filename)

    if last_id:
        next_id = generate_next_id(last_id)
    else:
        next_id = 'aaa0000'
    
    # Get the current time in GMT
    gmt_timezone = pytz.timezone('GMT')
    current_time_gmt = datetime.now(gmt_timezone)
    timestamp = current_time_gmt.strftime("%Y-%m-%d %H:%M:%S %Z")

    #Getting the doc name
    doc_name = PDF_PATH.split('/')[-1]

    # Check if the file exists
    file_exists = False
    try:
        with open(csv_filename, 'r') as file:
            file_exists = True
    except FileNotFoundError:
        pass
            
    # Write the new entry to the CSV file
    with open(csv_filename, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        #Write header if the file is new
        if not file_exists:
            # Assuming data contains the header
            writer.writerow(['Unique Identifier', 'Timestamp', 'Document Name'])
            
        writer.writerow([next_id, timestamp,  doc_name])


# Function to extract all the images
def extract_images_from_pdf(pdf_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page
    for page_number in range(pdf_document.page_count):
        # Get the page
        page = pdf_document[page_number]

        # Get the image list from the page
        image_list = page.get_images(full=True)

        # Iterate through each image on the page
        for image_index, img in enumerate(image_list):
            image_index += 1

            # Get the image information
            base_image = pdf_document.extract_image(img[0])
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Use Pillow to open and save the image
            image = Image.open(io.BytesIO(image_bytes))
            image_path = f"{output_folder}/page_{page_number + 1}_image_{image_index}.{image_ext}"
            image.save(image_path)

    # Close the PDF file
    pdf_document.close()

# Extracting paragraphs given word/sentence
def get_paragraph(word, PDF_PATH):
    y_diff = defaultdict(int)
    elements = []
    y_prev = 0
    for page_layout in extract_pages(PDF_PATH):
        for i,element in enumerate(page_layout):
            if isinstance(element, LTTextContainer):
                elements.append(element)
                x0, y0, x1, y1 = element.bbox
                if y_prev:
                    y_diff[int(y_prev-y1)]+=1
                y_prev = y0
                
    y_diff = {key: value for key, value in y_diff.items() if (value > 2 and key>0)}
    # Sort the keys based on values
    sorted_y_diff = sorted(y_diff, key=y_diff.get, reverse = True)
    final_text = ""
    skip_next = False
    for i in range(len(elements)-1, -1, -1):
        text = re.sub(r'\s+', ' ', elements[i].get_text())
        if re.sub(r'\s+',' ',word) in text and not skip_next and len(text)> len(word)+10:
            final_text = text+final_text
            x0, y0, x1, y1 = elements[i].bbox
            if i>=1:
                x3, y3, x4, y4 = elements[i-1].bbox
                if abs(y3-y1) < (sorted_y_diff[0]+sorted_y_diff[1])/2:
                    final_text = elements[i-1].get_text()[:-1]+final_text
                    skip_next = True
                else:
                    final_text = '\n'+final_text
        else:
            skip_next = False
            
    return final_text

#Function to extract paragraph relevant to tables
def ext_tab_rel_para(PDF_PATH):
    # Instantiation of OCR
    ocr = TesseractOCR(n_threads=1, lang="eng")

    # Instantiation of document, either an image or a PDF
    doc = PDF(PDF_PATH)

    # Table extraction
    extracted_tables = doc.extract_tables(ocr=ocr,
                                        implicit_rows=True,
                                        borderless_tables=True,
                                        min_confidence=5)

    paras = []
    for i in range(len(extracted_tables)):
        title = ''
        if extracted_tables[i]:
            title = extracted_tables[i][0].title
        else:
            paras.append('')
        # Regex pattern to extract entire expressions (case-insensitive)
        pattern = r'([tT]able[-\s]*[\d.-]+)'

        #Checking if atleast one table is found
        if title :
            # Find all matches in the text
            matches = re.findall(pattern, title)
            
            # Display the matched values
            for match in matches:
                if match[-1] == '.':
                    match = match[:-1]
                print("Matched Table Expressions:", match)
                para = get_paragraph(match, PDF_PATH)
                print('Page Number:- ', i+1)
                print('Paragraph:-', para)
                paras.append(para)
    
    return paras

#Function to filter stopwords
def remove_stopwords(sentence_list):
    stop_words = set(stopwords.words('english'))
    filtered_sentences = []
    for sentence in sentence_list:
        words = nltk.word_tokenize(sentence)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_sentence = ' '.join(filtered_words)
        filtered_sentences.append(filtered_sentence)
    return filtered_sentences

#Extracting text from stored images
def extract_txt_from_img(PDF_PATH):
    images_folder_path = "./extracted_data/"+PDF_PATH.split('/')[-1].split('.')[0] + '/images'
    paras = {}
    for img_name in os.listdir(images_folder_path):
        try:
            image_path = "./extracted_data/"+PDF_PATH.split('/')[-1].split('.')[0]+"/images/"+img_name
            text = pytesseract.image_to_string(image_path,lang='eng')
            text = text.strip().split('\n')
            text = [t for t in text if len(t)>0]
            text = list(set(text))
            text = remove_stopwords(text)
            text = [t for t in text if len(t.split())>1]
            for t in text:
                print("\nSearching for :" , t)
                print(get_paragraph(t, PDF_PATH))

            paras[img_name] = text
        except:
            pass
            # print("Error in working with :", img_name)

    return paras 