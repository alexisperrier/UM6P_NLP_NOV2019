'''
Yandex Language Modeling
https://github.com/yandexdataschool/nlp_course/blob/master/week03_lm/seminar.ipynb

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Alternative manual download link: https://yadi.sk/d/_nGyU2IajjR9-w
!wget "https://www.dropbox.com/s/99az9n1b57qkd9j/arxivData.json.tar.gz?dl=1" -O arxivData.json.tar.gz
!tar -xvzf arxivData.json.tar.gz
data = pd.read_json("./arxivData.json")
data.sample(n=5)

lines = data.apply(lambda row: row['title'] + ' ; ' + row['summary'], axis=1).tolist()

sorted(lines, key=len)[:3]

#  tokenization
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
lines = [' '.join(tokenizer.tokenize(line.lower())) for line in lines]


assert sorted(lines, key=len)[0] == \
    'differential contrastive divergence ; this paper has been retracted .'
assert sorted(lines, key=len)[2] == \
    'p = np ; we claim to resolve the p =? np problem via a formal argument for p = np .'


from tqdm import tqdm
from collections import defaultdict, Counter

# special tokens:
# - unk represents absent tokens,
# - eos is a special token after the end of sequence

UNK, EOS = "_UNK_", "_EOS_"

from nltk.util import ngrams
def count_ngrams(lines, n):
    counts = defaultdict(Counter)
    for line in sorted(lines, key=len):
        for gram in ngrams(line.split(),
                        n = n,
                        pad_right=True,
                        pad_left=True,
                        left_pad_symbol = UNK,
                        right_pad_symbol = EOS):
            counts[gram[:n-1]][ gram[n-1] ] += 1

    return counts

#  probabilities
class NGramLanguageModel:
    def __init__(self, lines, n):
        """
        Train a simple count-based language model:
        compute probabilities P(w_t | prefix) given ngram counts

        :param n: computes probability of next token given (n - 1) previous words
        :param lines: an iterable of strings with space-separated tokens
        """
        assert n >= 1
        self.n = n

        counts = count_ngrams(lines, self.n)
        # compute token proabilities given counts
        self.probs = defaultdict(Counter)
        # probs[(word1, word2)][word3] = P(word3 | word1, word2)
        for prefix, v in counts.items():
            total = sum( counts[prefix].values() )
            for w,c in v.items():
                self.probs[prefix][w] = c / total

    def get_possible_next_tokens(self, prefix):
        """
        :param prefix: string with space-separated prefix tokens
        :returns: a dictionary {token : it's probability} for all tokens with positive probabilities
        """
        prefix = prefix.split()
        prefix = prefix[max(0, len(prefix) - self.n + 1):]
        prefix = [ UNK ] * (self.n - 1 - len(prefix)) + prefix
        return self.probs[tuple(prefix)]

    def get_next_token_prob(self, prefix, next_token):
        """
        :param prefix: string with space-separated prefix tokens
        :param next_token: the next token to predict probability for
        :returns: P(next_token|prefix) a single number, 0 <= P <= 1
        """
        return self.get_possible_next_tokens(prefix).get(next_token, 0)


'''
Language model on whole corpus
'''

lm = NGramLanguageModel(lines, n=3)


def get_next_token(lm, prefix, temperature=1.0):
    """
    return next token after prefix;
    :param temperature: samples proportionally to lm probabilities ^ temperature
        if temperature == 0, always takes most likely token. Break ties arbitrarily.
    """
    next_tks = lm.get_possible_next_tokens(prefix)

    denominateur = sum( [ v** temperature for v in  next_tks.values() ] )

    candidates, probas = [], []
    for tok, prob in next_tks.items():
        candidates.append(tok)
        probas.append(prob ** temperature / denominateur)

    return np.random.choice(candidates, p = probas)


def perplexity(lm, lines, min_logprob=np.log(10 ** -50.)):
    """
    :param lines: a list of strings with space-separated tokens
    :param min_logprob: if log(P(w | ...)) is smaller than min_logprop,
        set it equal to min_logrob
    :returns: corpora-level perplexity
        - a single scalar number from the formula above

    Note: do not forget to compute P(w_first | empty) and P(eos | full_sequence)

    PLEASE USE lm.get_next_token_prob and NOT lm.get_possible_next_tokens
    """
    perplexity = 1
    N = 0
    for word in lines:
        N += 1
        tokens = list(lm.get_possible_next_tokens(word).keys())
        for token in tokens:
            perplexity = perplexity / lm.get_next_token_prob(word, token)
    perplexity = pow(perplexity, 1/float(N))

    return perplexity




# ---------------
