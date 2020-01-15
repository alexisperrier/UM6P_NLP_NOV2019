'''
Tokenization
'''


'''
Spacy
'''

import spacy
from spacy.lang.en import English
nlp = English()
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for token in doc:
    print(token.text)

'''
NLTK
https://www.nltk.org/api/nltk.tokenize.html
'''

from nltk.tokenize import sent_tokenize, word_tokenize

data = "All work and no play makes jack a dull boy, all work and no play"
print(word_tokenize(data))
data = "All work and no play makes jack a dull boy. All work and no play makes jack a dull boy."
print(sent_tokenize(data))


'''
Sentence tokenizer with NLTK punkt model

NLTK comes with tokenizers models in several languages.

    $> locate tokenizers



### Punkt Sentence Tokenizer

This tokenizer divides a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences. It must be trained on a large collection of plaintext in the target language before it can be used.

The NLTK data package includes a pre-trained Punkt tokenizer for English.
https://www.nltk.org/book/ch03.html

locate tokenizers
Punkt Sentence Tokenizer

'''
import nltk.data
text = '''
Punkt knows that the periods in Mr. Smith and Johann S. Bach
do not mark sentence boundaries.  And sometimes sentences
can start with non-capitalized words.  i is a good variable
name.
'''
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

print('\n-----\n'.join(sent_detector.tokenize(text.strip())))




# -------------
