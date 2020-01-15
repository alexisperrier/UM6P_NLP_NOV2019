'''
Text  Classification
Detect  Character in simpson dataset

- load dataset
- create dataset: Homer, Marge, Bart, Lisa
- remove all other characters

- clean data, pre process
- vectorize
- multinomial naive bayes

'''
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

df = pd.read_csv('../data/simpsons_dataset.csv').dropna()

characters = ['Homer Simpson', 'Marge Simpson', 'Bart Simpson', 'Lisa Simpson']
df = df[df.raw_character_text.isin(characters)]

df.raw_character_text.value_counts()

nlp = English()

'''
Tokenization et lemmatization
'''


df['word_count'] = df.spoken_words.apply(lambda text: len(text.split()) )

df = df[df.word_count > 10]

def lemmatize(text):
    doc = nlp(text)
    # premier passe pour lemmatizer et lowercase
    tokens = [token.lemma_.lower() for token in doc ]

    res = [ token for token in tokens
        if (token not in STOP_WORDS)
            & (token not in string.punctuation)
            & (token not in ['-pron-', '-','--'])
         ]
    return res


df['tokens'] = df.spoken_words.apply(lemmatize)

df['count_tokens'] = df.tokens.apply(len )

# --------------

df = df[df.count_tokens > 10]

df['text'] = df.tokens.apply( lambda txt : ' '.join(txt)   )
# -----------

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vectorizer = TfidfVectorizer(ngram_range = (1,5), min_df = 10)

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, auc, confusion_matrix, accuracy_score

model = MultinomialNB()

df = df.sample(frac = 1)

H = df[df.raw_character_text == 'Homer Simpson'].sample(500)

sampled_df = pd.concat( [df[df.raw_character_text != 'Homer Simpson'], H]    )

sampled_df = sampled_df.sample(frac = 1)

y = sampled_df.raw_character_text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform( sampled_df.text )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))





# --------------
