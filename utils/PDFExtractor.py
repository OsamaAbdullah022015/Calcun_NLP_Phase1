import PyPDF2
import pandas as pd
from pdfreader import PDFDocument
from img2table.ocr import TesseractOCR
from img2table.document import PDF
import csv


class PDFExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        text = ""
        with open(self.pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text
        print(text)
        return text

    def extract_sentences(self):
        sentences = []
        with open(self.pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                raw_sentences = page_text.split('. ')
                cleaned_sentences = [sentence.replace('\n', ' ') for sentence in raw_sentences]
                sentences.extend(cleaned_sentences)
        return sentences

    def extract_metadata(self):
        data = {}
        data['Source'] = self.pdf_path
        print("Source: ", self.pdf_path)
        pdf = PyPDF2.PdfReader(self.pdf_path)
        fd = open(self.pdf_path, "rb")
        doc = PDFDocument(fd)
        data["Number of pages"]= len(pdf.pages)
        print("Number of pages: ", len(pdf.pages))
        try: 
            if pdf.metadata.get('/Author'):
                data['Author'] = pdf.metadata['/Author'] 
                print("Author: ", pdf.metadata['/Author'])
            if pdf.metadata.get('/Creator'):
                data['Creator'] = pdf.metadata['/Creator']
                print("Creator: ", pdf.metadata['/Creator'])
            if pdf.metadata.get('/CreationDate'): 
                # data['Creation Date'] = pdf.metadata['/CreationDate']
                print("Creation Date: ", pdf.metadata['/CreationDate'])
            if pdf.metadata.get('/ModDate'):
                # data['ModDate'] = pdf.metadata['/ModDate']
                print("ModDate: ", pdf.metadata['/ModDate'])
            if pdf.metadata.get('/Producer'):
                # data['Producer'] = pdf.metadata['/Producer']
                print("Producer: ", pdf.metadata['/Producer'])
        except:
            print("Not possible to extract meta data from the pdf")
            
        page = next(doc.pages())
        print("There are", len(sorted(page.Resources.Font.keys())), "types of fonts in the given pdf document")
        fonts = ""
        for i, key in enumerate(sorted(page.Resources.Font.keys())):
            # data['For font key '+str(i+1)] = key
            print("For font key: ", key)
            font = page.Resources.Font[key]
            # data['Font Sub Type '+str(i+1)] = font.Subtype
            print("   Font Sub Type: ", font.Subtype)
            # data['Base Font '+str(i+1)] = font.BaseFont
            fonts += ','+str(font.BaseFont) if i != 0 else str(font.BaseFont) 
            print("   Base Font: ", font.BaseFont)
            # data['Font Encoding '+str(i+1)] = font.Encoding
            print("   Font Encoding: ", font.Encoding)

        print(fonts)
        data['Fonts'] = fonts
        for key, value in data.items():
            if not isinstance(value, (list, tuple)):
                data[key] = [value]
        
        # Assuming 'data' is a dictionary
        csv_file_path = './metadata/'+self.pdf_path.split('/')[-1].split('.')[0]+'.csv'

        try:
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                # Define the CSV columns based on the keys in the 'data' dictionary
                fieldnames = list(data.keys())
                
                # Create a CSV writer
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write the header
                writer.writeheader()
                
                # Write the data
                for row in zip(*data.values()):
                    writer.writerow(dict(zip(data.keys(), row)))
                
            print(f'Data has been saved to {csv_file_path}')
        except Exception as e:
            print(f'Error: {e}')
            

    def extract_key_value_pairs(self):
        tesseract_ocr = TesseractOCR(n_threads=1, lang="eng")
        print('.'+self.pdf_path.split('.')[1]+'.xlsx')
        pdf = PDF(src=self.pdf_path)
        try:
            pdf.to_xlsx('.'+self.pdf_path.split('.')[1]+'.xlsx',
                        ocr=tesseract_ocr,
                        implicit_rows=False,
                        borderless_tables=False,
                        min_confidence=50)
            # Read all sheets from the Excel file into a dictionary of data frames
            all_sheets = pd.read_excel('.'+self.pdf_path.split('.')[1]+'.xlsx', sheet_name=None)

            # Create a list of data frames from the dictionary
            tables = list(all_sheets.values())
            print(tables[0].content.values())
            s = [df.columns for df in tables]
            for df in tables:
                df.fillna('', inplace=True)
                
            # return tables
            # tables = tabula.read_pdf(self.pdf_path, pages='all')
            tables_dict = [tables[i].to_dict(orient='records') for i in range(len(tables))]
            return tables_dict, tables
        except:
            return None, None