# # Importing libraries

from IPython.utils import io
with io.capture_output() as captured:
    import spacy
    import re
    import os
    import argparse
    from gensim.utils import simple_preprocess
    from PyPDF2 import PdfReader
    from utils.Detectron import Detectron
    from utils.API import API
    from utils.PDFExtractor import PDFExtractor
    from utils.GraphPlotter import GraphPlotter
    from utils.TextProcessor import TextProcessor
    from utils.UploadFile import UploadFile
    from utils.ERExtractor import ERExtractor

import warnings
warnings.filterwarnings("ignore")


def handle_args():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument("--pdf_location", type=str, help="Input pdf location", required=True)
    parser.add_argument("--model_name", type=str, help="Input model name from ['en_core_web_sm', 'en_core_web_trf', 'en_core_web_lg']", required=True)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the arguments
    PDF_PATH = args.pdf_location
    MODEL = args.model_name

    # Use the arguments in your script
    print(f"PDF path is : {PDF_PATH}")
    print(f"Model selected is : {MODEL}")

    return PDF_PATH, MODEL

def main():

    PDF_PATH, MODEL = handle_args()

    baseURL = 'http://localhost:8000/app'
    nlp = spacy.load(MODEL)
    nlp.max_length = 1500000 #or any large value, as long as you don't run out of RAM

    # detectron = Detectron(PDF_PATH)
    pdf_extractor = PDFExtractor(PDF_PATH)
    text_processor = TextProcessor(pdf_extractor.extract_sentences())
    er_extractor = ERExtractor(MODEL)

    # # Extracting using Detectron2
    # # Detecting
    # detections = []
    # pdf = PdfReader(PDF_PATH)
    # for i in range(len(pdf.pages)):
    #   detected_layout, page_image = detectron.detect_and_draw_boxes(i)
    #   detections.append(detected_layout)

    # # Extracting
    # final_text = ""
    # for i in range(len(pdf.pages)):
    #     print('Page: ', i+1, '\n')
    #     detectron.extract(detections[i],i)
    # -

    ## Finding Key - Value Pairs
    key_value_pairs, tables = pdf_extractor.extract_key_value_pairs()
    if tables:
        tables[0].head()

    ##Creating Data Frame for ploting graph
    if key_value_pairs:
        table_dtf = er_extractor.key_value_pair_er_extractor(key_value_pairs)

    ## Plotting graphs for key value pairs
    # Plotting entity based graphs and saving them
    if tables and not table_dtf.empty:
        key_val_plotter = GraphPlotter(table_dtf, baseURL)
        tb_entities = table_dtf['entity'].unique()
        for entity in tb_entities:
            # Define a regular expression pattern to match sentences with no English characters
            non_english_pattern = r'^[^A-Za-z]*$'
            if not re.search(non_english_pattern, entity):
                key_val_plotter.plot_entity_based_graphs(entity)

    # # Extracting text from page
    txt = text_processor.replace_words_with_synonyms()
    print(txt)

    doc = nlp(txt)
    lst_docs = [sent for sent in doc.sents]
    print("Total Number of Sentences", len(lst_docs))

    # # Extracting Metadata
    pdf_extractor.extract_metadata()# Save it in a csv with name same as name of pdf

    # # Trying Topic Modelling
    os.environ['TOKENIZERS_PARALLELISM']='true'
    max_num_topics = 10
    topic_model, corpus, id2word = text_processor.get_topics(max_num_topics)

    # # Visualizing topics
    print(text_processor.visualize_topics(topic_model, corpus, id2word, is_notebook=0))

    # # Extracting entities and relations (ML approach)
    # Create a set of stop words 
    stop_words = spacy.lang.en.stop_words.STOP_WORDS

    dtf = er_extractor.txt_er_extractor(txt)
    dtf.head()

    # # Removing stopwords from the entities and creating a list of main topic words
    topic_words = set()
    for topic in topic_model.print_topics():
        for word in topic[1].split(' + '):
            topic_words.add(word.split('*')[1][1:-1])
            
    tmp = dtf['entity'].value_counts().to_dict()
    # print(topic_words)
    print(tmp)
    # Function to remove stopwords from a list of words
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) 
                if word not in stop_words] for doc in texts]
            
    filtered_words = []
    for word in remove_stopwords(list(tmp.keys())):
        filtered_words.extend(word)
        
    # Remove the stop words and creating dict
    entity_dict = {}
    for word in filtered_words:
        if word in tmp.keys():
            entity_dict[word] = tmp[word]

    topic_words = list(topic_words)

    # # Ploting KG for topic words and other entites
    # Plotting topic based graphs and saving them
    graph_plotter = GraphPlotter(dtf, baseURL)
    graph_plotter.plot_topic_based_graphs(topic_words, entity_dict,5)
    return 

if __name__ == "__main__":
    main()


