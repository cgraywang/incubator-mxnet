# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from .base import _StepwiseSeq2SeqModel, get_rnn_layer, get_rnn_cell, ExtendedSequential
from ... import nn
from .... import init


class AWDLSTM(_StepwiseSeq2SeqModel):
    def __init__(self, mode, vocab, embed_dim, hidden_dim, num_layers,
                 dropout=0.5, drop_h=0.5, drop_i=0.5, drop_e=0.1, weight_drop=0,
                 tie_weights=False, **kwargs):
        self._mode = mode
        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout
        self._drop_h = drop_h
        self._drop_i = drop_i
        self._drop_e = drop_e
        self._weight_drop = weight_drop
        self._tie_weights = tie_weights
        super(AWDLSTM, self).__init__(vocab, vocab, **kwargs)

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            if self._drop_e:
                pass # TODO weight drop
            embedding.add(nn.Embedding(len(self._in_vocab), self._embed_dim,
                                       weight_initializer=init.Uniform(0.1)))
            if self._drop_i:
                embedding.add(nn.Dropout(self._drop_i, axes = (0,))) # TODO variational drop
        return embedding

    def _get_encoder(self):
        encoder = ExtendedSequential()
        with encoder.name_scope():
            for l in range(self._num_layers):
                encoder.add(get_rnn_layer(self._mode, 1, self._embed_dim if l == 0 else
                                          self._hidden_dim, self._hidden_dim if
                                          l != self._num_layers - 1 or not self._tie_weights
                                          else self._embed_dim, 0, self._weight_drop))
                if self._drop_h:
                    pass # TODO variational drop
        return encoder

    def _get_decoder(self):
        vocab_size = len(self._out_vocab)
        if self._tie_weights:
            output = nn.Dense(vocab_size, flatten=False, params=self.embedding.params)
        else:
            output = nn.Dense(vocab_size, flatten=False)
        return output

    def begin_state(self, *args, **kwargs):
        return self.encoder[0].begin_state(*args, **kwargs)

class RNNModel(_StepwiseSeq2SeqModel):

    def __init__(self, mode, vocab, embed_dim, hidden_dim,
                 num_layers, dropout=0.5, tie_weights=False, **kwargs):
        self._mode = mode
        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._dropout = dropout
        self._tie_weights = tie_weights
        super(RNNModel, self).__init__(vocab, vocab, **kwargs)

    def _get_embedding(self):
        embedding = nn.HybridSequential()
        with embedding.name_scope():
            embedding.add(nn.Embedding(len(self._in_vocab), self._embed_dim,
                                       weight_initializer=init.Uniform(0.1)))
            if self._dropout:
                embedding.add(nn.Dropout(self._dropout, axes = (0,)))
        return embedding

    def _get_encoder(self):
        encoder = ExtendedSequential()
        with encoder.name_scope():
            for l in range(self._num_layers):
                encoder.add(get_rnn_layer(self._mode, 1, self._embed_dim if l == 0 else
                                          self._hidden_dim, self._hidden_dim if
                                          l != self._num_layers - 1 or not self._tie_weights
                                          else self._embed_dim, 0, 0))
    
        return encoder

    def _get_decoder(self):
        vocab_size = len(self._out_vocab)
        if self._tie_weights:
            output = nn.Dense(vocab_size, flatten=False, params=self.embedding.params)
        else:
            output = nn.Dense(vocab_size, flatten=False)
        return output

    def begin_state(self, *args, **kwargs):
        return self.encoder[0].begin_state(*args, **kwargs)
