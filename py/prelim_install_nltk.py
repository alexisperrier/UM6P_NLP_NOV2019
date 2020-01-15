'''
Install NLTK
conda install -c anaconda nltk
'''

import nltk
nltk.download('popular')
'''
NLTK Datasets
https://www.nltk.org/book/ch02.html
'''

'''
Gutenberg
'''
# load original text from Gutenberg texts
alice = nltk.corpus.gutenberg.open('carroll-alice.txt').read()

'''
Movie reviews
'''

from nltk.corpus import movie_reviews


# Brown corpus: 1971 million words annotated
from nltk.corpus import brown
brown.categories()
brown.open('cr09').read()
