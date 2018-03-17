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

from ... import Block, HybridBlock, Parameter, contrib, nn, rnn
from .... import nd


class _StepwiseSeq2SeqModel(Block):
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


def _apply_weight_drop_to_rnn_cell(block, rate, weight_dropout_mode = 'training'):
    params = block.collect_params('.*_h2h_weight')
    for key, value in params.items():
        weight_dropped_params = WeightDropParameter(value, rate, weight_dropout_mode)
        block.collect_params('.*_h2h_weight')._params[key] = weight_dropped_params
        for child_block in block._children:
            child_block.collect_params('.*_h2h_weight')._params[key] = weight_dropped_params

def get_rnn_cell(mode, num_layers, num_embed, num_hidden,
                 dropout, weight_dropout,
                 var_drop_in, var_drop_state, var_drop_out, weight_dropout_mode = 'training'):
    rnn_cell = rnn.SequentialRNNCell()
    with rnn_cell.name_scope():
        for i in range(num_layers):
            if mode == 'rnn_relu':
                cell = rnn.RNNCell(num_hidden, 'relu', input_size=num_embed)
            elif mode == 'rnn_tanh':
                cell = rnn.RNNCell(num_hidden, 'tanh', input_size=num_embed)
            elif mode == 'lstm':
                cell = rnn.LSTMCell(num_hidden, input_size=num_embed)
            elif mode == 'gru':
                cell = rnn.GRUCell(num_hidden, input_size=num_embed)
            if var_drop_in + var_drop_state + var_drop_out != 0:
                cell = contrib.rnn.VariationalDropoutCell(cell,
                                                          var_drop_in,
                                                          var_drop_state,
                                                          var_drop_out)

            rnn_cell.add(cell)
            if i != num_layers - 1 and dropout != 0:
                rnn_cell.add(rnn.DropoutCell(dropout))

            if weight_dropout:
                _apply_weight_drop(rnn_cell, weight_dropout, weight_dropout_mode = weight_dropout_mode)

    return rnn_cell

def _apply_weight_drop(block, rate, weight_dropout_mode = 'training', axes = ()):
    for k, v in block.params._params.items():
        if 'h2h_weight' in k:
            weight_dropped_params = WeightDropParameter(v, rate, weight_dropout_mode, axes)
            params_lst = []
            _retrieve_params(block, k, params_lst)
            for params in params_lst:
                params[k] = weight_dropped_params
    
def _retrieve_params(block, name, params_lst):
    if name in block.params._params:
        params_lst.append(block.params._params)
    for c_block in block._children:
        _retrieve_params(c_block, name, params_lst)
        
        
def _apply_weight_drop_to_rnn_layer_hack(block, rate, weight_dropout_mode = 'training'):
    b_params = block.params._params
    for key, val in b_params.items():
        if 'h2h_weight' in key:
            d = val._check_and_get(val._data, ctx = None)
            d = nd.Dropout(nd.array(val._data[0]), rate, weight_dropout_mode)
            if val._data is not None:
                d = nd.Dropout(val._data[0], rate, weight_dropout_mode, axes=(0,))
                b_params[key].set_data(d)      
    for child_block in block._children:
        _apply_weight_drop_to_rnn_layer_hack(child_block, rate, weight_dropout_mode)

        
def _apply_weight_drop_to_rnn_layer_old(block, rate, weight_dropout_mode = 'training'):
    params = block.collect_params('.*_h2h_weight')
    for key, value in params.items():
        weight_dropped_params = WeightDropParameter(value, rate, weight_dropout_mode)
        params._params[key] = weight_dropped_params
    for child_block in block._children:
        _apply_weight_drop_to_rnn_layer_old(child_block, rate, weight_dropout_mode)


def get_rnn_layer(mode, num_layers, num_embed, num_hidden, dropout, weight_dropout, weight_dropout_mode = 'training'):
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
    if weight_dropout:
        _apply_weight_drop(block, weight_dropout, weight_dropout_mode)

    return block


class RNNCellLayer(Block):
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


class WeightDropParameter(Parameter):
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
    def __init__(self, parameter, rate=0.0, mode='training', axes=()):
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
            d = nd.Dropout(d, self._rate, self._mode, self._axes)
        return d

    def __repr__(self):
        s = 'WeightDropParameter {name} (shape={shape}, dtype={dtype}, rate={rate}, mode={mode})'
        return s.format(name=self.name, shape=self.shape, dtype=self.dtype,
                        rate=self._rate, mode=self._mode)
