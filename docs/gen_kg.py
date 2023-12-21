# Importing libraries
import spacy
import PyPDF2
import networkx as nx
import textacy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tabula
import nltk
import warnings
import gensim
import json
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import re
import pdf2image
import psycopg2
import os
import requests
import layoutparser as lp
from IPython.display import SVG, display
from PIL import Image
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from pdfreader import PDFDocument
from nltk.corpus import wordnet, stopwords
from gensim.utils import simple_preprocess
from PyPDF2 import PdfReader
import pandas as pd
import spacy
import textacy

warnings.filterwarnings("ignore")

# Loading model
nlp = spacy.load('en_core_web_sm')

# Downloading required libraries
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

# Adding stopwards
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


class API:
    """Class for making API calls.

    This class provides methods for making POST and GET requests to a specified base URL.

    :param baseURL: The base URL for API requests.
    """
    def __init__(self, baseURL):
        """Initialize the API class with the base URL.

        :param baseURL: The base URL for API requests.
        """
        self.baseURL = baseURL

    def post(self, route, payload):
        """Send a POST request to the API and push the provided JSON payload to the specified route.

        :param route: The API route to send the request to.
        :param payload: JSON payload to send in the request.
        """
        # A POST request to the API
        post_response = requests.post(self.baseURL+'/'+route, json=payload)
        
        # Print the response
        post_response_json = post_response.json()
        print(post_response_json)

    def get(self, route):
        """Send a GET request to the API, fetch data from the specified route, and return the JSON response.

        :param route: The API route to retrieve data from.
        :return: The JSON response from the GET request.
        """
        # A GET request to the API
        response = requests.get(self.baseURL+'/'+route)
        
        # Print the response
        response_json = response.json()
        print(response_json)
        return response_json


class Detectron:
    """Class for working with the Detectron model for document layout analysis.

    This class can be used to detect text and layout elements in a PDF document.

    :param pdf_path: The path to the PDF document to analyze.
    """
    def __init__(self, pdf_path):
        """Initialize the Detectron class with the path to a PDF document.

        :param pdf_path: The path to the PDF document to extract information from.
        """
        self.pdf_path = pdf_path
        # Reading the PDF and creating Images
        self.doc = pdf2image.convert_from_path(pdf_path, dpi=300)

    # Function to detect text
    def detect_and_draw_boxes(self, page_index=0):
        """Detect layout elements on the specified page and draw bounding boxes around them.

        :param page_index: The index of the page to analyze, defaults to 0.
        :return: The detected layout elements and an image with bounding boxes.
        """
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
        """Extract text, figures, and tables from the detected layout of the PDF document.

        :param detected: The detected layout.
        :param i: The index of the page to extract from, defaults to 0.
        """
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
                dic_predicted[str(block.id)+"-"+block.type] = pd.read_csv( io.StringIO(extracted) )
            except:
                pass
        
        # Displaying the doc
        parse_doc(dic_predicted)
    
    # Funtion to Filter Non English Characters
    def filter_non_english(self, text):
        """Filter non-English characters from the input text.

        :param text: The input text.
        :return: The filtered text with non-English characters removed.
        """
        # Define a regular expression pattern to match non-English characters
        non_english_pattern = re.compile(r'[^\x00-\x7F]+')
    
        # Use the sub() method to replace non-English characters with an empty string
        filtered_text = non_english_pattern.sub('', text)
    
        return filtered_text


class PDFExtractor:
    """Class for extracting text, sentences, metadata, and key-value pairs from a PDF document.

    :param pdf_path: The path to the PDF document to extract information from.
    """
    def __init__(self, pdf_path):
        """Initialize the PDFExtractor class with the path to a PDF document.

        :param pdf_path: The path to the PDF document to extract information from.
        """
        self.pdf_path = pdf_path

    def extract_text(self):
        """Extract text from the PDF document.

        :return: The extracted text as a string.
        """
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
        """Extract sentences from the PDF document.

        :return: A list of sentences.
        """
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
        """Extract metadata information from the PDF document and print it.

        :return: None
        """
        print("Source: ", self.pdf_path)
        pdf = PyPDF2.PdfReader(self.pdf_path)
        fd = open(self.pdf_path, "rb")
        doc = PDFDocument(fd)
        print("Number of pages: ", len(pdf.pages))
        print("Author: ", pdf.metadata['/Author'])
        print("Creator: ", pdf.metadata['/Creator'])
        print("Creation Date: ", pdf.metadata['/CreationDate'])
        print("ModDate: ", pdf.metadata['/ModDate'])
        print("Producer: ", pdf.metadata['/Producer'])
        page = next(doc.pages())
        print("There are", len(sorted(page.Resources.Font.keys())), "types of fonts in the given pdf document")
        for key in sorted(page.Resources.Font.keys()):
            print("For font key: ", key)
            font = page.Resources.Font[key]
            print("   Font Sub Type: ", font.Subtype)
            print("   Base Font: ", font.BaseFont)
            print("   Font Encoding: ", font.Encoding)

    def extract_key_value_pairs(self):
        """Extract key-value pairs from tables in the PDF document.

        :return: A list of dictionaries representing tables.
        """
        tables = tabula.read_pdf(self.pdf_path, pages='all')
        tables_dict = [tables[i].to_dict(orient='records') for i in range(len(tables))]
        return tables_dict, tables


class GraphPlotter(API):
    """Class for plotting knowledge graphs based on entities and relations.

    :param dtf: A pandas DataFrame containing entity-relation-object triples.
    :param baseURL: The base URL for API requests.
    """
    def __init__(self, dtf, baseURL):
        """Initialize the GraphPlotter class with a DataFrame and a base URL for API requests.

        :param dtf: A pandas DataFrame containing entity-relation-object triples.
        :param baseURL: The base URL for API requests.
        """
        super().__init__(baseURL)
        self.dtf = dtf

    def plot_graph(self, entity):
        """Plot a knowledge graph centered around the specified entity.

        :param entity: The central entity to build the graph around.
        """
        tmp = self.dtf[(self.dtf["entity"]==entity) | (self.dtf["object"]==entity)]
        G = nx.from_pandas_edgelist(tmp, source="entity", target="object",
                                    edge_attr="relation",
                                    create_using=nx.DiGraph())
        plt.figure(figsize=(15,10))
        pos = nx.spring_layout(G, k=1)
        node_color = ["red" if node==entity else "skyblue" for node in G.nodes]
        edge_color = ["red" if edge[0]==entity else "black" for edge in G.edges]
        nx.draw(G, pos=pos, with_labels=True, node_color=node_color,
                edge_color=edge_color, cmap=plt.cm.Dark2,
                node_size=2000, node_shape="o", connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_edge_labels(G, pos=pos, label_pos=0.5,
                                      edge_labels=nx.get_edge_attributes(G,'relation'),
                                      font_size=12, font_color='black', alpha=0.6)
        
        self.saving_kg_as_json(entity)
        plt.show()
        
    def plot_topic_based_graphs(self, topic_words, entity_dict, max_graphs):
        """Plot knowledge graphs based on topic words and entity frequencies.

        :param topic_words: A list of topic words.
        :param entity_dict: A dictionary of entities and their frequencies.
        :param max_graphs: The maximum number of graphs to plot.
        """
        topic_entities = []
        for word in topic_words:
            if entity_dict.get(word,0) >= 2:
                topic_entities.append(word)
        if topic_entities:
            for word in topic_entities:
                self.plot_graph(word)
        if len(topic_entities) < max_graphs:
            cnt = 0
            for word, count in entity_dict.items():
                if word in topic_entities:
                    continue
                self.plot_graph(word)
                cnt += 1
                if cnt+len(topic_entities) == max_graphs:
                    break
    
    def plot_entity_based_graphs(self, word):
        """Plot a knowledge graph centered around a specific entity.

        :param word: The entity to build the graph around.
        """
        self.plot_graph(word)

    def saving_kg_as_json(self, entity):
        """Save a knowledge graph as JSON and send it via a POST request to the API.

        :param entity: The entity for which the knowledge graph is saved.
        """
        tmp = self.dtf[(self.dtf["entity"]==entity) | (self.dtf["object"]==entity)]
        G = nx.from_pandas_edgelist(tmp, source="entity", target="object",
                                    edge_attr="relation",
                                    create_using=nx.DiGraph())
        
        # Creating a dict of nodes
        dict_nodes = {}
        for i, node in enumerate(G.nodes):
            dict_nodes[node] = i+1
        
        #Finding nodes
        nodes = []
        for key,val in dict_nodes.items():
            nodes.append({'id':val, 'name':key})
        
        #Finding edges and relation
        links = []
        for key, val in nx.get_edge_attributes(G, 'relation').items():
            links.append({ "source": dict_nodes[key[0]], "target": dict_nodes[key[1]],  "desc": val })
        
        #Building kg
        kg = {
            'nodes':nodes,
            'links':links
        }

        clean_entity = re.sub(r'[^a-zA-Z ]', '', entity)
    
        # Split the sentence into words
        words = clean_entity.split()
    
        # If there is only one word, return it as is
        result = ""
        if len(words) == 1:
            result = words[0].lower()
        else:
            # Take the first three characters of each word and convert them to lowercase
            first_three_chars = [word[:3].lower() for word in words]
            # Join the words with underscores
            result = '_'.join(first_three_chars)
    
        payload = {
            'data' : kg,
            'file_name' : result
        }
        self.post('save-json', payload)
            

class TextProcessor:
    """Class for processing text and extracting topics.

    :param sentences: A list of sentences for text processing.
    """
    def __init__(self, sentences):
        """Initialize the TextProcessor class with a list of sentences.

        :param sentences: A list of sentences for text processing.
        """
        self.sentences = sentences

    def get_topics(self, num_topics = 10):
        """Extract topics from the text using LDA.
        Using Perplexity score for finding the ideal number of topics.
        Perplexity score measures how well a probability distribution predicts a sample, often used in natural language processing with lower scores indicating better models.
        :param num_topics: The number of topics to extract, defaults to 10.
        :return: The LDA model, corpus, and id2word dictionary.
        """
        data_words = list(self.sent_to_words(self.sentences))
        
        # remove stop words
        data_words = self.remove_stopwords(data_words)
        
        # Create Dictionary
        id2word = corpora.Dictionary(data_words)
            
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in data_words]
        
        # Build LDA model
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics)
        
        # Print the Keyword in the 10 topics
        print(lda_model.print_topics())
        doc_lda = lda_model[corpus]
        
        return lda_model, corpus, id2word

    def replace_words_with_synonyms(self):
        """Replace words in the text with synonyms based on Word2Vec and K-means clustering. Used silhouette score for finding the ideal size of clusters.  Silhouette score assesses the cohesion and separation of clusters in data, with higher scores indicating better-defined clusters in clustering analysis.

        :return: The text with replaced synonyms.
        """
        input_text = ' '.join(self.sentences)
        tokens = nltk.word_tokenize(input_text)
        synonyms = defaultdict(list)
        for token in tokens:
            synsets = wordnet.synsets(token)
            if synsets:
                synonyms[token].extend([lemma.name() for synset in synsets for lemma in synset.lemmas()])
        synonyms_list = [synonym for synonyms_set in synonyms.values() for synonym in synonyms_set]
        word2vec_model = Word2Vec([synonyms_list], vector_size=100, window=5, min_count=1, sg=1)
        word_vectors = [word2vec_model.wv[word] for word in synonyms_list]
        num_clusters = 15
        kmeans = KMeans(n_clusters=num_clusters)
        clusters = kmeans.fit_predict(word_vectors)
        cluster_representatives = {}
        for cluster_idx in range(num_clusters):
            cluster_indices = [i for i, cluster_label in enumerate(clusters) if cluster_label == cluster_idx]
            representative_idx = cluster_indices[0]
            representative_word = synonyms_list[representative_idx]
            for idx in cluster_indices:
                synonyms_list[idx] = representative_word
        output_text = ' '.join([synonyms_list[tokens.index(token)] if token in synonyms_list else token for token in tokens])
        return output_text

    def sent_to_words(self, sentences):
        """Tokenize sentences into words and remove stopwords.

        :param sentences: A list of sentences.
        :return: A generator of tokenized words.
        """
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

    # Function to remove stopwords from a list of words
    def remove_stopwords(self, texts):
        """Remove stopwords from a list of words.

        :param texts: A list of tokenized words.
        :return: A list of words with stopwords removed.
        """
        return [[word for word in simple_preprocess(str(doc)) 
                 if word not in stop_words] for doc in texts]

    def visualize_topics(self, topic_model, corpus, id2word):
        """Visualize topics using pyLDAvis.

        :param topic_model: The LDA model.
        :param corpus: The corpus of documents.
        :param id2word: The dictionary of id to word mapping.
        :return: Visualization data for topics.
        """
        pyLDAvis.enable_notebook()
        LDAvis_prepared = gensimvis.prepare(topic_model, corpus, id2word)
        return LDAvis_prepared


class ERExtractor:
    def __init__(self) -> None:
        """
        Initialize an Entity-Relation Extractor.

        This class provides methods for extracting key-value pairs and entities and relations
        from text data.

        """
        pass

    def key_value_pair_er_extractor(self,key_value_pairs):
        """
        Extract entity-relation data from a list of key-value pairs.

        Args:
            key_value_pairs (list of dict): A list of dictionaries containing key-value pairs.

        Returns:
            pandas.DataFrame: A DataFrame with columns 'entity', 'relation', and 'object' containing
            the extracted entity-relation data.

        """
        table_dtf = pd.DataFrame()
        for tab in key_value_pairs:
            table = pd.DataFrame(tab)
            for col in table.columns:
                dtf = pd.DataFrame()
                dtf['entity']=[col]*len(tab)
                dtf['relation'] = ''
                dtf['object'] = table[col].tolist()
                table_dtf = pd.concat([table_dtf, dtf], ignore_index=True)

        table_dtf.dropna(inplace = True)

        return table_dtf
        
    def txt_er_extractor(self,txt):
        """
        Extract entity-relation data from a text using NLP processing.

        Args:
            txt (str): The input text for entity-relation extraction.

        Returns:
            pandas.DataFrame: A DataFrame with columns 'id', 'text', 'entity', 'relation', and 'object'
            containing the extracted entity-relation data.

        """
        dic = {"id":[], "text":[], "entity":[], "relation":[], "object":[]}
        doc = nlp(txt)

        lst_docs = [sent for sent in doc.sents]
                
        for n,sentence in enumerate(lst_docs):
            lst_generators = list(textacy.extract.subject_verb_object_triples(sentence)) 
            # print(lst_generators)
            for sent in lst_generators:
                subj = "_".join(map(str, sent.subject))
                obj  = "_".join(map(str, sent.object))
                relation = "_".join(map(str, sent.verb))
                dic["id"].append(n)
                dic["text"].append(sentence.text)
                dic["entity"].append(subj)
                dic["object"].append(obj)
                dic["relation"].append(relation)


        #Creating dataframe
        dtf = pd.DataFrame(dic)
        dtf['entity'] = dtf['entity'].str.lower()
        dtf['relation'] = dtf['relation'].str.lower()

        return dtf
        

