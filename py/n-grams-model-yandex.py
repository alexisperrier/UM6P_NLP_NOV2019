'''
n-grams model
https://github.com/yandexdataschool/nlp_course/blob/master/week03_lm/seminar.ipynb
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


!wget "https://www.dropbox.com/s/99az9n1b57qkd9j/arxivData.json.tar.gz?dl=1" -O arxivData.json.tar.gz
!tar -xvzf arxivData.json.tar.gz
data = pd.read_json("./arxivData.json")
data.sample(n=5)

lines = data.apply(lambda row: row['title'] + ' ; ' + row['summary'], axis=1).tolist()
sorted(lines, key=len)[:3]

'''
# Task: convert lines (in-place) into strings of space-separated tokens. import & use WordPunctTokenizer
'''

lines = [  ' '.join( [ tk.lower() for tk in    tokenizer.tokenize(line)]) for line in lines]

assert sorted(lines, key=len)[0] == 'differential contrastive divergence ; this paper has been retracted .'
assert sorted(lines, key=len)[2] == 'p = np ; we claim to resolve the p =? np problem via a formal argument for p = np .'



from tqdm import tqdm
from collections import defaultdict, Counter

# special tokens:
# - unk represents absent tokens,
# - eos is a special token after the end of sequence
lines = sorted(lines, key=len)[:100]
UNK, EOS = "_UNK_", "_EOS_"

def count_ngrams(lines, n):
    """
    Count how many times each word occured after (n - 1) previous words
    :param lines: an iterable of strings with space-separated tokens
    :returns: a dictionary { tuple(prefix_tokens): {next_token_1: count_1, next_token_2: count_2}}

    When building counts, please consider the following two edge cases
    - if prefix is shorter than (n - 1) tokens, it should be padded with UNK.
    For n=3,
      empty prefix: "" -> (UNK, UNK)
      short prefix: "the" -> (UNK, the)
      long prefix: "the new approach" -> (new, approach)
    - you should add a special token, EOS, at the end of each sequence
      "... with deep neural networks ." -> (..., with, deep, neural, networks, ., EOS)
      count the probability of this token just like all others.
    """

    counts = defaultdict(Counter)
    for line in dummy_lines:
        for w1, w2, w3 in ngrams(line.split(' '), n, pad_right=True, pad_left=True, left_pad_symbol = UNK, right_pad_symbol  = EOS):
            counts[(w1, w2)][w3] += 1

    # counts[(word1, word2)][word3] = how many times word3 occured after (word1, word2)

    return counts



dummy_lines = sorted(lines, key=len)[:100]
dummy_counts = count_ngrams(dummy_lines, n=3)
assert set(map(len, dummy_counts.keys())) == {2}, "please only count {n-1}-grams"
assert len(dummy_counts[('_UNK_', '_UNK_')]) == 78
assert dummy_counts['_UNK_', 'a']['note'] == 3
assert dummy_counts['p', '=']['np'] == 2
assert dummy_counts['author', '.']['_EOS_'] == 1



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

        # populate self.probs with actual probabilities
        for ngrm in counts:
            total_count = float(sum(counts[ngrm].values()))
            for w in counts[ngrm]:
                counts[ngrm][w] /= total_count
                # <YOUR CODE>

    def get_possible_next_tokens(self, prefix):
        """
        :param prefix: string with space-separated prefix tokens
        :returns: a dictionary {token : it's probability}
            for all tokens with positive probabilities
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


# --------------------------------
