import networkx as nx
import re
import os
import rdflib
from rdflib import URIRef, Literal
from rdflib.namespace import RDF, RDFS
from rdflib.plugins.serializers.nt import NTSerializer
import urllib.parse
import matplotlib.pyplot as plt
from utils.API import API

class GraphPlotter(API):
    def __init__(self, dtf, baseURL):
        super().__init__(baseURL)
        self.dtf = dtf

    def get_valid_uri(self, node):
        # Replace spaces with underscores and encode special characters
        return URIRef(urllib.parse.quote(node.replace(" ", "_")))

    def save_graph_as_rdf(self, graph, entity):
        # Create an RDF graph
        rdf_graph = rdflib.Graph()
    
        # Iterate through the nodes and add them as RDF resources
        for node in graph.nodes:
            subject = self.get_valid_uri(node)
            rdf_graph.add((subject, RDF.type, RDFS.Class))
    
        # Iterate through the edges and add them as RDF triples
        for edge in graph.edges:
            subject = self.get_valid_uri(edge[0])
            predicate = self.get_valid_uri(edge[1])
            relation = Literal(graph.edges[edge]['relation'])
            rdf_graph.add((subject, predicate, relation))
    
        # Serialize the RDF graph to RDF/XML format
        rdf_file_path = os.path.join("output", f"{entity}.rdf")
        with open(rdf_file_path, "wb") as rdf_file:
            serializer = NTSerializer(rdf_graph)
            serializer.serialize(rdf_file)
    
        print(f"Saved RDF graph for {entity} in {rdf_file_path}")
    
    def plot_graph(self, entity):
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
        self.save_graph_as_rdf(G, entity)
        # plt.savefig('./output/'+entity+'.svg', format='SVG')
        # self.write_blob('svgtable', entity+'.svg')
        # os.remove('./output/'+entity+'.svg')
        plt.show()
        
    def plot_topic_based_graphs(self, topic_words, entity_dict, max_graphs):
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
        self.plot_graph(word)

    def saving_kg_as_json(self, entity):
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

        # Saving the JSON data in DB
        # self.write_blob('JSONTable', entity+'.json',json.dumps(kg))
        # Remove special characters and spaces from the sentence
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
        # self.post('save-json', payload)
            