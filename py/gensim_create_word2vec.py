'''
Word Embedding creation with Gensim
https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296
'''

import nltk
# corpora > Brown and Corpora Conll200
# nltk.download()
from nltk.corpus import brown
import string
from gensim.models import Word2Vec
from spacy.lang.en.stop_words import STOP_WORDS
import multiprocessing

sentences = brown.sents()

'''
Parameters
- sentences
- size : the size of the embedding (lower -> better for POS, higher -> better for other tasks)
- window: window of context words
- min_count: ignore rare words
- negative: number of negative samples
- iter: number of epochs
- workers
'''

embed_dim = 50

w2v = Word2Vec(sentences,
        size = embed_dim,
        window = 3,
        min_count = 5,
        workers = 8,
        negative = 15,
        iter = 5)

# similarity

'''
word embedding on Simpsons
'''

from nltk.tokenize import word_tokenize
df = pd.read_csv('./data/simpsons_dataset.csv')
df.head()

df.dropna(subset=['spoken_words'] , inplace = True)

# keep only nouns, adj and verbs

def pos_filter(text):
    doc = nlp(text)
    tokens = []
    for tk in doc:
        if tk.pos_ in ['NOUN','VERB','PROPN','ADJ']:
            tokens.append(tk.lemma_.lower().strip()  )
    return tokens

df['tokens'] = df.spoken_words.apply(pos_filter)

simp_sentences = df.tokens.values

w2v = Word2Vec(simp_sentences,
        size = embed_dim,
        window = 3,
        min_count = 5,
        workers = 8,
        negative = 15,
        iter = 5)

w2v2 = Word2Vec(simp_sentences,
        sg = 1,
        size = 128,
        window = 5,
        min_count = 5,
        workers = 8,
        negative = 15,
        iter = 10)

'''
Calculate some words distance
'''
w2v2.most_similar('bread')

'''
For each word in the vocabulary, get the vector, the count and the word
'''
# ex: access vectors
w2v2.wv.get_vector('job')
# ex: vocab as a dict
w2v2.wv.vocab['homer'].count

# build a dataframe
vect = []

stop_words = set(list(STOP_WORDS) + ['-pron-','be','are',"we're",'do'])
for word in w2v2.wv.vocab.keys():
    if (word not in string.punctuation + '...') & (len(word) > 2)& (word not in stop_words):
        vect.append({
            'word': word,
            'n': w2v2.wv.vocab[word].count,
            'v': w2v2.wv.get_vector(word)
        })
vect = pd.DataFrame(vect)
vect.shape
vect.n.describe()

# remove words that are too frequent or too rare
vect = vect[(vect.n > 7) & (vect.n < 4500)]
vect.sort_values(by = 'n', ascending = False, inplace = True)



# T-SNE

from sklearn.manifold import TSNE
nword = 200
data = vect[:nword]
words = vect[:nword].word.values
tsne = TSNE(n_components=2,
        perplexity= 40,
        verbose = 1,
        learning_rate = 50,
        n_iter = 5000
    )
X = tsne.fit_transform(list(data.v.values))

fig, ax = plt.subplots(1,1, figsize=(10,7))
plt.scatter(X[:, 0], X[:, 1], alpha = 0.5)

for n in range(nword):
    plt.text(X[n, 0], X[n, 1], words[n], fontsize=8)

plt.tight_layout()
plt.show()



'''
compare with Glove
see https://medium.com/@japneet121/word-vectorization-using-glove-76919685ee0b
'''

# pip install glove_python

# Then:

from glove import Corpus, Glove

#Creating a corpus object
corpus = Corpus()

#Training the corpus to generate the co occurence matrix which is used in GloVe
corpus.fit(lines, window=10)

glove = Glove(no_components=5, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove.model')

'''
FastText ?
'''



# ------------------
