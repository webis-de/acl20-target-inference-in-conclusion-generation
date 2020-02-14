import nltk
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence, StackedEmbeddings
from scipy.spatial import distance
import numpy as np
import fasttext
from nltk.corpus import stopwords
from flair.embeddings import BertEmbeddings


# initialize the word embeddings

# bert_embedding = DocumentPoolEmbeddings([BertEmbeddings()])
# glove_embedding = DocumentPoolEmbeddings([WordEmbeddings('glove')])

#bert_embedding = BertEmbeddings()
#glove_embedding = WordEmbeddings('glove')

# flair_embedding_forward = FlairEmbeddings('news-forward')
# flair_embedding_backward = FlairEmbeddings('news-backward')
# stacked_embeddings = StackedEmbeddings([glove_embedding,
#                                       flair_embedding_backward,
#                                       flair_embedding_forward])

nltk.download('stopwords')

embedding_sizes = {'glove': 100, 'bert': 3072, 'flair_stacked': 5000, 'fasttext':300}

embedding_methods = {
    #'glove' : glove_embedding,
    #'bert' : bert_embedding,
    'fasttext' : None
}

print('Loading fasttext....')
fasttext_model = fasttext.load_model("/home/miladalshomary/Development/data/crawl-300d-2M-subword/crawl-300d-2M-subword.bin")

grammar = r"""
            NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
            VP:
            {<V.*>}  # terminated with Verbs
            NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """


cp = nltk.RegexpParser(grammar)

english_stopwords = stopwords.words('english')

def extract_nps(pos_text):
    pos_text = list(map(lambda x: (x['text'], x['type']), pos_text))
    #print(pos_text)
    parsed_text = cp.parse(pos_text)
    
    nps = []
    for subtree in parsed_text.subtrees():
        if subtree.label() == 'NP':
            nps.append(subtree.leaves())

    return nps

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def embed_sentence(target_phrase, target_context, normalize=True, embedding_method_name='glove'):
    embedding_size   = embedding_sizes[embedding_method_name]
    embedding_method = embedding_methods[embedding_method_name]

    if embedding_method_name == 'fasttext':
        #to lower
        target_phrase = target_phrase.lower()
        #remove stop words
        target_phrase_tokens = [x for x in target_phrase.split(' ') if x not in english_stopwords]
        vectors = [fasttext_model[token] for token in target_phrase_tokens]

        if len(target_phrase_tokens) == 0:
            print('WARNING: trying to embed empty list of tokens.. returning default random vector')
            return np.random.uniform(0,1, size=embedding_size).astype(np.float32)

        if normalize:
            vectors = [vec/np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else np.zeros(embedding_size, dtype=np.float32) for vec in vectors]

        avg_embedding = np.mean(vectors, axis=0)

        return avg_embedding

    if embedding_method_name == 'glove': #no need for contextual info
        target_phrase = target_phrase.lower() #Lower case all tokens..
        sentence = Sentence(target_phrase)
    else:
        sentence = Sentence(target_context)
    
    
    embedding_method.embed(sentence)



    tokens = sentence.tokens
    
    if embedding_method_name == 'bert': # Average only the tokens of the target phrase
        target_phrase_tokens  = target_phrase.split(' ')
        target_context_tokens = target_context.split(' ')
        # print(target_phrase_tokens)
        # print(target_context_tokens)
        #Take only tokens of the target phrase
        tokens = [x for x in sentence.tokens if x.text in target_phrase_tokens]
    
    #print(len(tokens))
    if len(tokens) == 0:
        print('WARNING: trying to embed empty list of tokens.. returning default random vector')
        return np.random.uniform(0,1, size=embedding_size).astype(np.float32)

    #average the embdedding of the tokens
    vectors = [token.embedding.numpy() for token in tokens]
    if normalize:
        vectors = [vec/np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else np.zeros(embedding_size, dtype=np.float32) for vec in vectors]

    avg_embedding = np.mean(vectors, axis=0)

    #if the the resulted average is vector of zeros then make it random... To avoide cases of nan when computing cosine distance
    if np.all(avg_embedding == 0):
        print('WARNING: All tokens in the sentence where unkown.. returning default random vector')
        return np.random.uniform(0,1, size=embedding_size).astype(np.float32)
    else:
        return avg_embedding