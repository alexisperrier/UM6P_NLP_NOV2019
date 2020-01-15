'''
Normalization
'''



'''
Lemmatization
'''

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# add tokenization
[lemmatizer.lemmatize(word) for word in text.split(' ')]
['caress', 'fly', 'dy', 'mule', 'denied',
 'died', 'agreed', 'owned', 'humbled', 'sized',
 'meeting', 'stating', 'siezing', 'itemization',
 'sensational', 'traditional', 'reference', 'colonizer',
 'plotted']

# Spacy
doc = nlp(' '.join(words))
>>> [token.lemma_ for token in doc]
['caress', 'fly', 'die', 'mule', 'deny',
 'die', 'agree', 'own', 'humble', 'sized',
 'meeting', 'state', 'siezing', 'itemization',
 'sensational', 'traditional', 'reference', 'colonizer',
 'plot']



'''
Example on some text
'''
# is the tokenziation before the  lemmatization of after?
# same time for spacy?

text = '''Playing dumb, I asked one of them, “Who’s Michael Jordan?” They kinda stared, then one offered, “Some guy. He cries a lot.”'''
