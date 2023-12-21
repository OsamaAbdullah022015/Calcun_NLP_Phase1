import pandas as pd
import spacy
import textacy


class ERExtractor:
    def __init__(self, MODEL) -> None:
        """
        Initialize an Entity-Relation Extractor.

        This class provides methods for extracting key-value pairs and entities and relations
        from text data.

        """
        self.nlp = spacy.load(MODEL)
        self.nlp.max_length = 1500000 #or any large value, as long as you don't run out of RAM

        
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
        doc = self.nlp(txt)

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
        