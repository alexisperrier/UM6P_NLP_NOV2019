'''
Topic Modeling LSA
'''

import pandas as pd
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English


df = pd.read_csv('../data/techno.csv')

# clean  up content

def clean_text(text):
    # enlever les html
    text = re.sub("<[^>]*>",' ', text)
    # enlever chiffres
    text = re.sub("[\d]",' ', text)
    # enlever chiffres
    # text = text.replace("\n",'')
    text = re.sub("[\n]+",' ', text)
    text = re.sub("[-]+",' ', text)
    text = re.sub("[\s]+",' ', text)
    return text

def remove_punctuation(text):
    all_punct = string.punctuation + "—”“…®"
    return re.sub(r"[{}]".format(all_punct),'',text)

def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_.lower().strip() for token in doc ]

def filter(tokens):
    return [ token.strip() for token in tokens
        if (token not in STOP_WORDS)
            & len(token) > 2

            & (token not in ['-pron-','\ufeff1'])
         ]


df['clean_content'] = df.content.apply(clean_text)
df['clean_content'] = df.clean_content.apply(remove_punctuation)
df['tokens'] = df.clean_content.apply(lemmatize)
df['tokens'] = df.tokens.apply(filter)

df['article'] = df.tokens.apply(lambda tks : ' '.join(tks))

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# vectorizer = TfidfVectorizer(stop_words='english',use_idf=True,smooth_idf=True)
# svd_model = TruncatedSVD(n_components=100,    algorithm='randomized',n_iter=10)
#
# X = vectorizer.fit_transform(df.article)
# svd_model.fit_transform(svd_model)





# -----
