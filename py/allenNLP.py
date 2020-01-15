'''
AllenNLP from the book Real World NLP
'''

from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.trainer import Trainer

from realworldnlp.predictors import SentenceClassifierPredictor

EMBEDDING_DIM = 128
HIDDEN_DIM = 128

# wget https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip

reader = StanfordSentimentTreeBankDatasetReader()
train_dataset = reader.read('/home/alexis/NLP/data/trees/train.txt')
dev_dataset = reader.read('/home/alexis/NLP/data/trees/dev.txt')

'''
# vocabulary
Vocabulary class that manages mappings from some linguistic units (such as characters, words, and labels) to their IDs. You can tell the class to create a Vocabulary instance from a set of instances:
'''
vocab = Vocabulary.from_instances(train_dataset + dev_dataset,  min_count={'tokens': 5})

vocab.print_statistics()
vocab.get_vocab_size()


'''
Embedding
'''
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=EMBEDDING_DIM)

#  use word_embeddings to convert words to their embeddings.
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})


'''
RNN LSTM
'''
encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
# Linear layer

linear = torch.nn.Linear(in_features=encoder.get_output_dim(),out_features=vocab.get_vocab_size('labels'))

'''
Model
'''

class LstmClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary,
                 positive_label: str = '4') -> None:
        super().__init__(vocab)
        # We need the embeddings to convert word IDs to their vector representations
        self.word_embeddings = word_embeddings

        self.encoder = encoder

        # After converting a sequence of vectors to a single vector, we feed it into
        # a fully-connected linear layer to reduce the dimension to the total number of labels.
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))

        # Monitor the metrics - we use accuracy, as well as prec, rec, f1 for 4 (very positive)
        positive_index = vocab.get_token_index(positive_label, namespace='labels')
        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(positive_index)

        # We use the cross entropy loss because this is a classification task.
        # Note that PyTorch's CrossEntropyLoss combines softmax and log likelihood loss,
        # which makes it unnecessary to add a separate softmax layer.
        self.loss_function = torch.nn.CrossEntropyLoss()

    # Instances are fed to forward after batching.
    # Fields are passed through arguments with the same name.
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        # In deep NLP, when sequences of tensors in different lengths are batched together,
        # shorter sequences get padded with zeros to make them equal length.
        # Masking is the process to ignore extra zeros added by padding
        mask = get_text_field_mask(tokens)

        # Forward pass
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)

        # In AllenNLP, the output of forward() is a dictionary.
        # Your output dictionary must contain a "loss" key for your model to be trained.
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_measure(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self.f1_measure.get_metric(reset)
        return {'accuracy': self.accuracy.get_metric(reset),
                'precision': precision,
                'recall': recall,
                'f1_measure': f1_measure}

iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
iterator.index_with(vocab)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

trainer = Trainer(model = model,
    optimizer = optimizer,
    iterator=iterator,
    train_dataset=train_dataset,
    validation_dataset=dev_dataset,
    num_epochs=20)

trainer.train()

'''
Skip gram on AllenNLP
'''
import math
import random
from collections import Counter

import numpy as np
import torch
import torch.optim as optim
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField
from allennlp.data.instance import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.training.trainer import Trainer
from overrides import overrides
from scipy.stats import spearmanr
from torch.nn import CosineSimilarity
from torch.nn import functional

EMBEDDING_DIM = 256
BATCH_SIZE = 256
CUDA_DEVICE = 0

'''
from https://github.com/mhagiwara/realworldnlp/blob/master/examples/embeddings/word2vec.py
'''

@DatasetReader.register("skip_gram")
class SkipGramReader(DatasetReader):
    def __init__(self, window_size=5, lazy=False, vocab: Vocabulary=None):
        """A DatasetReader for reading a plain text corpus and producing instances
        for the SkipGram model.
        When vocab is not None, this runs sub-sampling of frequent words as described
        in (Mikolov et al. 2013).
        """
        super().__init__(lazy=lazy)
        self.window_size  = window_size
        self.reject_probs = None
        if vocab:
            self.reject_probs = {}
            threshold = 1.e-3
            token_counts = vocab._retained_counter['token_in']  # HACK
            total_counts = sum(token_counts.values())
            for _, token in vocab.get_index_to_token_vocabulary('token_in').items():
                counts = token_counts[token]
                if counts > 0:
                    normalized_counts = counts / total_counts
                    reject_prob = 1. - math.sqrt(threshold / normalized_counts)
                    reject_prob = max(0., reject_prob)
                else:
                    reject_prob = 0.
                self.reject_probs[token] = reject_prob

    def _subsample_tokens(self, tokens):
        """Given a list of tokens, runs sub-sampling.
        Returns a new list of tokens where rejected tokens are replaced by Nones.
        """
        new_tokens = []
        for token in tokens:
            reject_prob = self.reject_probs.get(token, 0.)
            if random.random() <= reject_prob:
                new_tokens.append(None)
            else:
                new_tokens.append(token)

        return new_tokens

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), "r") as text_file:
            for line in text_file:
                tokens = line.strip().split(' ')
                tokens = tokens[:1000000]  # TODO: remove

                if self.reject_probs:
                    tokens = self._subsample_tokens(tokens)
                    print(tokens[:200])  # for debugging

                for i, token in enumerate(tokens):
                    if token is None:
                        continue

                    token_in = LabelField(token, label_namespace='token_in')

                    for j in range(i - self.window_size, i + self.window_size + 1):
                        if j < 0 or i == j or j > len(tokens) - 1:
                            continue

                        if tokens[j] is None:
                            continue

                        token_out = LabelField(tokens[j], label_namespace='token_out')
                        yield Instance({'token_in': token_in, 'token_out': token_out})




reader = SkipGramReader()

text8 = reader.read('some pure text file')

vocab = Vocabulary.from_instances(text8, min_count={'token_in': 5, 'token_out': 5})
iterator = BasicIterator(batch_size=BATCH_SIZE)
iterator.index_with(vocab)
embedding_in = Embedding(num_embeddings=vocab.get_vocab_size('token_in'),embedding_dim=EMBEDDING_DIM)


#A AllenNLP requires every model is inherited from Mode
class SkipGramModel(Model):
    def __init__(self, vocab, embedding_in):
        super().__init__(vocab)
        #B Embedding object is passed from outside rather than defined inside
        self.embedding_in = embedding_in
        #C This creates a linear layer (note that we don’t need biases)
        self.linear = torch.nn.Linear(
            in_features = EMBEDDING_DIM,
            out_features = vocab.get_vocab_size('token_out'),
            bias = False)

    #D Body of neural network computation is implemented in forward()
    def forward(self, token_in, token_out):
        #E Convert input tensors (word IDs) to word embeddings
        embedded_in = self.embedding_in(token_in)    #E
        # F apply the linear layer
        logits = self.linear(embedded_in)    #F
        # compute the loss
        loss = functional.cross_entropy(logits, token_out)    #G
        return {'loss': loss}





# -------------------------
