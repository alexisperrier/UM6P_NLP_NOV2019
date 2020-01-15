'''
Zipf's law
'''

from collections import Counter
import string

nltk.download('reuters')

from nltk.corpus import reuters

# reuters.words()
reuters_words = [w.lower() for w in reuters.words()
    if w not  in string.punctuation]

print(len(reuters_words))

cwords = Counter(reuters_words)
cwords.most_common(10)

import matplotlib.pyplot as plt
n = 40
# list of words
rw = [ s[0] for s in  cwords.most_common(n)]
# frequency of words
rv = [ s[1] / len(reuters_words) * 100.0 for s in  cwords.most_common(n)]
# zipf's law
rz = [rv[0] / (n+1) for n in range(n)     ]


fig, ax = plt.subplots(1,1, figsize= (12,5))
plt.plot(rw,rv, label = 'corpus')
plt.plot(rw,rz, label = "Zipf's law")
plt.legend()
plt.grid(alpha = 0.3)
plt.title("Zipf's law on Reuters corpus")


'''
Simpson's corpus
'''
import pandas as pd
from nltk.tokenize import word_tokenize
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


df = pd.read_csv('../data/simpsons_dataset.csv')
df.dropna( inplace = True)
text = ' '.join(df.spoken_words.values)
tokens = word_tokenize(text)

tokens = [tk.lower() for tk in tokens if tk not in string.punctuation ]
swords = Counter(tokens)


vs = [ s[1] / len(tokens) * 100.0 for s in  swords.most_common(n)]
ws = [ s[0] for s in  swords.most_common(n)]
zs = [vs[0] / (n+1) for n in range(n)     ]

fig, ax = plt.subplots(1,1, figsize= (12,5))
plt.plot(ws,vs, label = 'Simpsons corpus')
plt.plot(ws,zs, label = "Zipf's law")
plt.legend()
plt.grid(alpha = 0.3)
plt.title("Zipf's law on Simpsons corpus")
