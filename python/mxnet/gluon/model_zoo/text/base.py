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

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import contrib, nn, rnn


class _StepwiseSeq2SeqModel(gluon.Block):
    def __init__(self, in_vocab, out_vocab, **kwargs):
        super(_StepwiseSeq2SeqModel, self).__init__(**kwargs)
        self._in_vocab = in_vocab
        self._out_vocab = out_vocab
        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        raise NotImplementedError

    def _get_encoder(self):
        raise NotImplementedError

    def _get_decoder(self):
        raise NotImplementedError

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

    def forward(self, inputs, begin_state=None):
        embedded_inputs = self.embedding(inputs)
        if not begin_state:
            begin_state = self.begin_state()
        encoded, state = self.encoder(embedded_inputs, begin_state)
        out = self.decoder(encoded)
        return out, state
    
    
def _apply_weight_drop_to_rnn_cell(block, rate, mode = 'training'):
    params = block.collect_params('h2h_weight')
    weight_dropped_params = WeightDropParameter(params['h2h_weight'], rate, mode)
    block.params['h2h_weight'] = weight_dropped_params
    
    

def get_rnn_cell(mode, num_layers, num_hidden,
                 dropout, weight_dropout,
                 var_drop_in, var_drop_state, var_drop_out, training = True):
    rnn_cell = rnn.SequentialRNNCell()
    with rnn_cell.name_scope():
        for i in range(num_layers):
            if mode == 'rnn_relu':
                cell = rnn.RNNCell(num_hidden, 'relu')
            elif mode == 'rnn_tanh':
                cell = rnn.RNNCell(num_hidden, 'tanh')
            elif mode == 'lstm':
                cell = rnn.LSTMCell(num_hidden)
            elif mode == 'gru':
                cell = rnn.GRUCell(num_hidden)
            if var_drop_in + var_drop_state + var_drop_out != 0:
                cell = contrib.rnn.VariationalDropoutCell(cell,
                                                          var_drop_in,
                                                          var_drop_state,
                                                          var_drop_out)

            rnn_cell.add(cell)
            if i != num_layers - 1 and dropout != 0:
                rnn_cell.add(rnn.DropoutCell(dropout))
    
    if weight_dropout:
        if training:
            _apply_weight_drop_to_rnn_cell(rnn_cell, rate = weight_dropout, mode = 'training')
        else:
            _apply_weight_drop_to_rnn_cell(rnn_cell, rate = weight_dropout, mode = 'always')
    
    return rnn_cell


def _apply_weight_drop_to_rnn_layer(block, rate, mode = 'training'):
    params = block.collect_params('.*_h2h_weight')
    print("params.items()")
    print(params.items())
    for key, value in params.items():
        weight_dropped_params = WeightDropParameter(value, rate, mode)
        block.params._params[key] = weight_dropped_params
        for child_block in block._children:
            child_block.params._params[key] = weight_dropped_params
        
        block._unfused.params._params[key] = weight_dropped_params
        for cell_block in block._unfused:
            cell_block.params._params[key] = weight_dropped_params
        
#         block.params._shared._params[key] = weight_dropped_params
#     print("block.params._params.items()")
#     print(block.params._params.items())
#     print("block.params._shared")
#     print(block.params._shared)

#ignore bidirectional
def get_rnn_layer(mode, num_layers, num_embed, num_hidden, dropout, weight_dropout, training = True):
    if mode == 'rnn_relu':
        block = rnn.RNN(num_hidden, 'relu', num_layers, dropout=dropout,
                       input_size=num_embed)      
    elif mode == 'rnn_tanh':
        block = rnn.RNN(num_hidden, num_layers, dropout=dropout,
                       input_size=num_embed)
    elif mode == 'lstm':
        block = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                        input_size=num_embed)
    elif mode == 'gru':
        block = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                       input_size=num_embed)
#     if weight_dropout:
#         if training:
#             _apply_weight_drop_to_rnn_layer(block, rate = weight_dropout, mode = 'training')
#         else:
#             _apply_weight_drop_to_rnn_layer(block, rate = weight_dropout, mode = 'always')
    return block
    

class RNNCellLayer(gluon.Block):
    """A block that takes an rnn cell and makes it act like rnn layer."""
    def __init__(self, rnn_cell, layout='TNC', **kwargs):
        super(RNNCellBlock, self).__init__(**kwargs)
        self.cell = rnn_cell
        assert layout == 'TNC' or layout == 'NTC', \
            "Invalid layout %s; must be one of ['TNC' or 'NTC']"%layout
        self._layout = layout
        self._axis = layout.find('T')
        self._batch_axis = layout.find('N')

    def forward(self, inputs, states=None):
        batch_size = inputs.shape[self._batch_axis]
        skip_states = states is None
        if skip_states:
            states = self.cell.begin_state(batch_size, ctx=inputs.context)
        if isinstance(states, ndarray.NDArray):
            states = [states]
        for state, info in zip(states, self.cell.state_info(batch_size)):
            if state.shape != info['shape']:
                raise ValueError(
                    "Invalid recurrent state shape. Expecting %s, got %s."%(
                        str(info['shape']), str(state.shape)))
        states = sum(zip(*((j for j in i) for i in states)), ())
        outputs, states = self.cell.unroll(
            inputs.shape[self._axis], inputs, states,
            layout=self._layout, merge_outputs=True)

        if skip_states:
            return outputs
        return outputs, states

class ExtendedSequential(nn.Sequential):
    def forward(self, *x):
        for block in self._children:
            x = block(*x)
        return x


class WeightDropParameter(gluon.Parameter):
    """A Container holding parameters (weights) of Blocks and performs dropout.
    parameter : Parameter
        The parameter which drops out.
    rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
        Dropout is not applied if dropout_rate is 0.
    mode : str, default 'training'
        Whether to only turn on dropout during training or to also turn on for inference.
        Options are 'training' and 'always'.
    """
    def __init__(self, parameter, rate=0.0, mode='training'):
        p = parameter
        super(WeightDropParameter, self).__init__(
                name=p.name, grad_req=p.grad_req, shape=p._shape, dtype=p.dtype,
                lr_mult=p.lr_mult, wd_mult=p.wd_mult, init=p.init,
                allow_deferred_init=p._allow_deferred_init,
                differentiable=p._differentiable)
        self._rate = rate
        self._mode = mode

    def data(self, ctx=None):
        """Returns a copy of this parameter on one context. Must have been
        initialized on this context before.
        Parameters
        ----------
        ctx : Context
            Desired context.
        Returns
        -------
        NDArray on ctx
        """
        d = self._check_and_get(self._data, ctx)
        if self._rate:
            d = nd.Dropout(d, self._rate, self._mode)
        return d
