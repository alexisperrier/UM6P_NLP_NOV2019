'''
Langage detector
'''

from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from overrides import overrides
from allennlp.common.file_utils import cached_path

class TatoebaSentenceReader(DatasetReader):
    def __init__(self,
        token_indexers: Dict[str, TokenIndexer]= None,
        lazy = False):

        super().__init__(lazy = lazy)
        # define tokenizer
        self.tokenizer = CharacterTokenizer()
        # define token_indexers
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self,tokens, label=None):
        fields = {}
        fields['tokens'] = TextField(tokens, self.token_indexers)
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    @overrides
    def _read(self,file_path):
        file_path = cached_path(file_path)
        with open(file_path, "r") as text_file:
            # tsv separated
            for line in text_file:
                lang_id, sent = line.rstrip().split('\t')

                tokens = self.tokenizer.tokenize(sent)
                yield self.text_to_instance(tokens, lang_id)


EMBEDDING_DIM = 16
HIDDEN_DIM    = 16
reader = TatoebaSentenceReader()
train_set = reader.read('/home/alexis/NLP/realworldnlp/data/tatoeba/sentences.top10langs.train.tsv')
dev_set = reader.read('/home/alexis/NLP/realworldnlp/data/tatoeba/sentences.top10langs.dev.tsv')

# vocab
vocab = Vocabulary.from_instances(train_set, min_count = {'tokens':3})
token_embedding = Embedding(
    num_embeddings = vocab.get_vocab_size('tokens'),
    embedding_dim = EMBEDDING_DIM
)
#
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

encoder = PytorchSeq2VecWrapper(
    torch.nn.LSTM(EMBEDDING_DIM,HIDDEN_DIM, batch_first = True)
)

model = LstmClassifier(word_embeddings, encoder, vocab, positive_label = 'eng')

iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
iterator.index_with(vocab)
optimizer = optim.Adam(model.parameters())
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_set,
                  validation_dataset=dev_set,
                  patience = 5,
                  num_epochs=10)
trainer.train()



# Predictions

def classify(text: str, model: LstmClassifier):
    tokenizer = CharacterTokenizer()
    token_indexers = {'tokens': SingleIdTokenIndexer()}

    tokens = tokenizer.tokenize(text)
    instance = Instance({'tokens': TextField(tokens, token_indexers)  })
    logits = model.forward_on_instance(instance)['logits']
    label_id = np.argmax(logits)
    label = model.vocab.get_token_from_index(label_id, 'labels')
    print("text: {} \nlabel: {}".format(text, label))
    return logits







# -----------------
