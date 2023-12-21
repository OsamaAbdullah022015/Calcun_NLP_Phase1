import pdf2image
import layoutparser as lp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from PIL import Image


class Detectron:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        # Reading the PDF and creating Images
        self.doc = pdf2image.convert_from_path(pdf_path, dpi=300)

    # Function to detect text
    def detect_and_draw_boxes(self, page_index=0):
        # Loading model
        layout_model = lp.models.Detectron2LayoutModel("lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
                                                       extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                                       label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})
    
        # Convert document page to numpy array
        page_image = np.asarray(self.doc[page_index])
    
        # Predict layout elements
        detected_elements = layout_model.detect(page_image)
    
        # Sort detected elements based on y-coordinate
        detected_elements.sort(key=lambda x: x.coordinates[1])
    
        # Assign new IDs to the sorted elements
        detected_layout = lp.Layout([block.set(id=idx) for idx, block in enumerate(detected_elements)])
    
        # Convert numpy array back to PIL Image and return detected layout
        return detected_layout, Image.fromarray(page_image)

    def extract(self, detected, i=0):
      '''
      {'0-Title': '...',
      '1-Text':  '...',
      '2-Figure': array([[ [0,0,0], ...]]),
      '3-Table': pd.DataFrame,
      }
      '''
      model = lp.TesseractAgent(languages='eng')
      dic_predicted = {}
      def parse_doc(dic):
          for k,v in dic.items():
              if "Title" in k:
                  print('\x1b[1;31m'+ v +'\x1b[0m')
              elif "Figure" in k:
                  plt.figure(figsize=(10,5))
                  plt.imshow(v)
                  plt.show()
              else:
                  print(v)
              print(" ")
    
      img = np.asarray(self.doc[i])
    
      # Extracting Text
      for block in [block for block in detected if block.type in ["Title","Text"]]:
          ## segmentation
          segmented = block.pad(left=15, right=15, top=5, bottom=5).crop_image(img)
          ## extraction
          extracted = model.detect(segmented)
          ## save
          dic_predicted[str(block.id)+"-"+block.type] = extracted.replace('\n',' ').strip()
    
      #Extracting Figures
      for block in [block for block in detected if block.type == "Figure"]:
          ## segmentation
          segmented = block.pad(left=15, right=15, top=5, bottom=5).crop_image(img)
          ## save
          dic_predicted[str(block.id)+"-"+block.type] = segmented
    
      #Extracting Tables
      for block in [block for block in detected if block.type == "Table"]:
          try:
            ## segmentation
            segmented = block.pad(left=15, right=15, top=5, bottom=5).crop_image(img)
            ## extraction
            extracted = model.detect(segmented)
            ## save
            dic_predicted[str(block.id)+"-"+block.type] = pd.read_csv(io.StringIO(extracted) )
          except:
            pass
    
      # Displaying the doc
      parse_doc(dic_predicted)

    # Funtion to Filter Non English Characters
    def filter_non_english(self, text):
        # Define a regular expression pattern to match non-English characters
        non_english_pattern = re.compile(r'[^\x00-\x7F]+')
    
        # Use the sub() method to replace non-English characters with an empty string
        filtered_text = non_english_pattern.sub('', text)
    
        return filtered_text