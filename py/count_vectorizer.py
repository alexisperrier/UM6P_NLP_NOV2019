'''
Count vectorizer
'''

df = pd.read_csv('../data/simpsons_dataset.csv').dropna()
df = df[df.raw_character_text == 'Homer Simpson']



from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df.spoken_words)

vectorizer.vocabulary_
vectorizer.get_feature_names()

# use min_df, Max_df, stop words, ... to reduce  the  vocabulary

from spacy.lang.en.stop_words import STOP_WORDS
vectorizer = CountVectorizer(stop_words = STOP_WORDS, min_df = 0.01 )
X = vectorizer.fit_transform(df.spoken_words)

X.shape
# (27850, 55)


'''
Hashing vectorizer
'''
from sklearn.feature_extraction.text import HashingVectorizer

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
    'This not the second document.'
]
vectorizer = HashingVectorizer(n_features=4)
X = vectorizer.fit_transform(corpus)
print(X.shape)
X.toarray()

'''
Tf-IdF
'''
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.spoken_words)


'''
Viz
'''

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

# Compare Marge and Homer

df = pd.read_csv('../data/simpsons_dataset.csv').dropna()

homer = df[df.raw_character_text == 'Homer Simpson']
marge = df[df.raw_character_text == 'Marge Simpson']
lisa  = df[df.raw_character_text == 'Lisa Simpson']
bart  = df[df.raw_character_text == 'Bart Simpson'].reset_index()

vectorizer = CountVectorizer(stop_words = STOP_WORDS, min_df = 20 )
# vectorizer = HashingVectorizer(n_features=64, stop_words = STOP_WORDS, norm = 'l1' )
vectorizer = TfidfVectorizer(stop_words = STOP_WORDS, min_df = 20)

H = vectorizer.fit_transform(homer.spoken_words)
M = vectorizer.fit_transform(marge.spoken_words)
L = vectorizer.fit_transform(lisa.spoken_words)
B = vectorizer.fit_transform(bart.spoken_words)

pca = PCA(n_components=2)
h = pca.fit_transform(H.toarray())
m = pca.fit_transform(M.toarray())
l = pca.fit_transform(L.toarray())
b = pca.fit_transform(B.toarray())

fig, ax = plt.subplots(2,2, figsize = (8,8))
plt.subplot(2,2,1)
plt.title("Homer")
plt.scatter(h[:,0], h[:,1], alpha = 0.3)

plt.subplot(2,2,2)
plt.title("Lisa")
plt.scatter(l[:,0], l[:,1], alpha = 0.3)

plt.subplot(2,2,3)
plt.title("Marge")
plt.scatter(m[:,0], m[:,1], alpha = 0.3)

plt.subplot(2,2,4)
plt.title("Bart")
plt.scatter(b[:,0], b[:,1], alpha = 0.3)


# ----------------------

from sklearn.metrics.pairwise import cosine_similarity

# word - word similarity based on doc vectors

vectorizer = TfidfVectorizer(stop_words = STOP_WORDS, min_df = 10, )
# vectorizer = HashingVectorizer(n_features=64, stop_words = STOP_WORDS, norm = 'l1' )
vectorizer = CountVectorizer(stop_words = STOP_WORDS, min_df = 10)

def top10(word,character, vectorizer):
    top = 15

    X = vectorizer.fit_transform(character.spoken_words)
    X = X.toarray()

    cossim = cosine_similarity( np.matrix.transpose(X) )

    vocab = vectorizer.get_feature_names()
    print("Vocab size: {}".format(len(vocab)))
    if word not in vocab:
        raise "word {} not in vocab".format(word)
    else:
        word_index = vocab.index(word)
        closest_words_idx = np.argpartition(cossim[word_index,:]  , -top)[-top:]
        res = []
        for idx in closest_words_idx:
            res.append( {'word_idx': idx, 'word':  vocab[idx], 'sim' : cossim[word_index,idx] } )

        res = pd.DataFrame(res)
        res.sort_values(by = 'sim', ascending = False, inplace = True)
        return res

# docs containing the word
# docs_vector = np.nonzero(B[:,word_index])
# lines containing the word
# bart.loc[docs_vector].spoken_words.values




sim = []
w = B[:,word_index]
vocab = vectorizer.get_feature_names()
for n in range(len(vocab)):
    v = B[:,n]
    sim.append({ 'word':vocab[n], 'csim':cosine_similarity(v,w)})






# -----------------
