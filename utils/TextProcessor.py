import gensim.corpora as corpora
import gensim
import nltk
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
# from collections import defaultdict
# from nltk.corpus import wordnet, stopwords
from nltk.corpus import stopwords
# from gensim.models import Word2Vec
# from sklearn.cluster import KMeans
from gensim.utils import simple_preprocess
# from sklearn.metrics import silhouette_score
import nlpaug.augmenter.word as naw

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

class TextProcessor:
    def __init__(self, sentences):
        self.sentences = sentences

    def get_topics(self, max_num_topics = 10):    # Easier and better approach using BERTopic (https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic)
        data_words = list(self.sent_to_words(self.sentences))
        
        # remove stop words
        data_words = self.remove_stopwords(data_words)
        
        # Create Dictionary
        id2word = corpora.Dictionary(data_words) # Creates id for every unique word
        
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in data_words] # Gives Bag of Words
        
        # Calculate model perplexity
        perplexity_values = []

        for num_topics in range(1, max_num_topics + 1):  # Try different numbers of topics
            if len(perplexity_values)>=2 and abs(perplexity_values[-1]-perplexity_values[-2])<0.03 and perplexity_values[-1]<perplexity_values[-2]:
                print(f"Final model with {len(perplexity_values)} topics and perplexity value as {perplexity_values[-1]}") 
                break 
            lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics, passes=10, iterations=15)

            # Calculate the perplexity for this model
            perplexity = lda_model.log_perplexity(corpus)
            perplexity_values.append(perplexity)

            print(f"Number of topics: {num_topics}, Perplexity: {perplexity}")
        
        # Print the Keyword in the 10 topics
        print(lda_model.print_topics())
        doc_lda = lda_model[corpus]
        
        return lda_model, corpus, id2word

    def replace_words_with_synonyms(self):
        input_text = ' '.join(self.sentences)
        caug = naw.ContextualWordEmbsAug(
        # option to choose from is "word2vec", "glove" or "fasttext"
        model_path='distilbert-base-uncased',

        # options available are insert or substitute
        action='substitute')
        # augmented text
        augmented_text = caug.augment(input_text,n=1)[0]

        return augmented_text

        # tokens = nltk.word_tokenize(input_text)
        # synonyms = defaultdict(list)
        # for token in tokens:
        #     synsets = wordnet.synsets(token)
        #     if synsets:
        #         synonyms[token].extend([lemma.name() for synset in synsets for lemma in synset.lemmas()])
        # synonyms_list = [synonym for synonyms_set in synonyms.values() for synonym in synonyms_set]
        # word2vec_model = Word2Vec([synonyms_list], vector_size=100, window=5, min_count=1, sg=1)
        # word_vectors = [word2vec_model.wv[word] for word in synonyms_list]
        # max_num_clusters = 31
        # silhouette_scores = []
        # for i in range(2, max_num_clusters):
        #     kmeans = KMeans(n_clusters=i)
        #     clusters = kmeans.fit_predict(word_vectors)
        #     score = silhouette_score(word_vectors, kmeans.labels_)
        #     print(f"Silhouette score for {i} clusters is {score}")
        #     silhouette_scores.append(score)
        # cluster_representatives = {}
        # num_clusters=silhouette_scores.index(max(silhouette_scores))+2
        # kmeans = KMeans(n_clusters=num_clusters)
        # clusters = kmeans.fit_predict(word_vectors)
        # score = silhouette_score(word_vectors, kmeans.labels_)
        # print(f"Silhouette score for final model with {num_clusters} clusters is {score}") 
        # for cluster_idx in range(num_clusters):
        #     cluster_indices = [i for i, cluster_label in enumerate(clusters) if cluster_label == cluster_idx]
        #     representative_idx = cluster_indices[0]
        #     representative_word = synonyms_list[representative_idx]
        #     for idx in cluster_indices:
        #         synonyms_list[idx] = representative_word
        # output_text = ' '.join([synonyms_list[tokens.index(token)] if token in synonyms_list else token for token in tokens])
        return output_text

    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield(simple_preprocess(str(sentence), deacc=True)) # Simple Preprocess does not remove stopwords

    # Function to remove stopwords from a list of words
    def remove_stopwords(self, texts):
        return [[word for word in simple_preprocess(str(doc)) 
                 if word not in stop_words] for doc in texts]

    def visualize_topics(self, topic_model, corpus, id2word, is_notebook = 1):
        if is_notebook:
            pyLDAvis.enable_notebook()
        LDAvis_prepared = gensimvis.prepare(topic_model, corpus, id2word)
        return LDAvis_prepared