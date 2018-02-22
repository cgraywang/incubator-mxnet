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

import argparse
import time
import math
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import contrib
from mxnet.gluon.model_zoo.text.lm import RNNModel, AWDLSTM

parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Language Model on Wikitext-2.')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=750,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropout_h', type=float, default=0.3,
                    help='dropout applied to hidden layer (0 = no dropout)')
parser.add_argument('--dropout_i', type=float, default=0.4,
                    help='dropout applied to input layer (0 = no dropout)')
parser.add_argument('--dropout_e', type=float, default=0.1,
                    help='dropout applied to embedding layer (0 = no dropout)')
parser.add_argument('--weight_dropout', type=float, default=0.65,
                    help='weight dropout applied to h2h weight matrix (0 = no weight dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--cuda', action='store_true',
                    help='Whether to use gpu')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model')
parser.add_argument('--gctype', type=str, default='none',
                    help='type of gradient compression to use, \
                          takes `2bit` or `none` for now.')
parser.add_argument('--gcthreshold', type=float, default=0.5,
                    help='threshold for 2bit gradient compression')
parser.add_argument('--eval_only', action='store_true',
                    help='Whether to only evaluate the trained model')
parser.add_argument('--num_gpus', type=int, default=1,
                    help='number of gpus')
args = parser.parse_args()


###############################################################################
# Load data
###############################################################################


print(args)


if args.cuda:
    context = [mx.gpu(i) for i in range(args.num_gpus)]
    print('Running on', context)
else:
    context = mx.cpu(0)

train_dataset = contrib.data.text.WikiText2('./data', 'train', seq_len=args.bptt)
vocab = train_dataset.vocabulary
val_dataset, test_dataset = [contrib.data.text.WikiText2('./data', segment,
                                                         vocab=vocab,
                                                         seq_len=args.bptt)
                             for segment in ['validation', 'test']]

nbatch_train = len(train_dataset) // args.batch_size
train_data = gluon.data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   sampler=contrib.data.IntervalSampler(len(train_dataset),
                                                                        nbatch_train),
                                   last_batch='discard')

nbatch_val = len(val_dataset) // args.batch_size
val_data = gluon.data.DataLoader(val_dataset,
                                 batch_size=args.batch_size,
                                 sampler=contrib.data.IntervalSampler(len(val_dataset),
                                                                      nbatch_val),
                                 last_batch='discard')

nbatch_test = len(test_dataset) // args.batch_size
test_data = gluon.data.DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  sampler=contrib.data.IntervalSampler(len(test_dataset),
                                                                       nbatch_test),
                                  last_batch='discard')


###############################################################################
# Build the model
###############################################################################


ntokens = len(vocab)

##debug
print("args.weight_dropout" + str(args.weight_dropout))

if args.weight_dropout:
    model = AWDLSTM(args.model, vocab, args.emsize, args.nhid, args.nlayers,
                 args.dropout, args.dropout_h, args.dropout_i, args.dropout_e, args.weight_dropout,
                 args.tied)
    print("awdlstm")
else:
    model = RNNModel(args.model, vocab, args.emsize, args.nhid,
                 args.nlayers, args.dropout, args.tied)
    print("rnnmodel")
    
#initialization
model.collect_params().initialize(mx.init.Xavier(), ctx=context)


compression_params = None if args.gctype == 'none' else {'type': args.gctype, 'threshold': args.gcthreshold}
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': args.lr,
                         'momentum': 0,
                         'wd': 0},
                        compression_params=compression_params)
loss = gluon.loss.SoftmaxCrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size, ctx=context)
    for i, (data, target) in enumerate(data_source):
        data = data.as_in_context(context).T
        target = target.as_in_context(context).T
        output, hidden = model(data, hidden)
        L = loss(mx.nd.reshape(output, (-3, -1)),
                 mx.nd.reshape(target, (-1, 1)))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

def train():
    best_val = float("Inf")
    start_train_time = time.time()
    
    #TODO: add debug infor
    print("args.epochs=" + str(args.epochs))
    
    for epoch in range(args.epochs):
        total_L = 0.0
        start_epoch_time = time.time()
#         hidden = model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size, ctx=context)
        
        hiddens = [model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size, ctx=ctx) for ctx in context]
        for i, (data, target) in enumerate(train_data):
            start_batch_time = time.time()
            
            data = data.T
            target= target.T
            
            data_list = gluon.utils.split_and_load(data, context, even_split=False)
            target_list = gluon.utils.split_and_load(target, context, even_split=False)
            
            hiddens = [detach(hidden) for hidden in hiddens]
            
            Ls = []
            with autograd.record():
                for X, y, h in zip(data_list, target_list, hiddens):
                    output, h = model(X, h)
                    Ls.append(loss(mx.nd.reshape(output, (-3, -1)), mx.nd.reshape(y, (-1, 1))))
            for L in Ls:
                L.backward()

            
            for ctx in context:
                grads = [p.grad(ctx) for p in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
                gluon.utils.clip_global_norm(grads, args.clip * args.bptt * args.batch_size)

            trainer.step(args.batch_size)
            total_L += sum([mx.nd.sum(L).asscalar() for L in Ls])
            
            #what does this do?????? why set total_L = 0.0 ...
#             if i % args.log_interval == 0 and i > 0:
#                 cur_L = total_L / args.bptt / args.batch_size / args.log_interval
#                 print('[Epoch %d Batch %d] loss %.2f, ppl %.2f'%(
#                     epoch, i, cur_L, math.exp(cur_L)))
#                 total_L = 0.0
            
#             print('[Epoch %d Batch %d] throughput %.2f samples/s'%(
#                     epoch, i, args.batch_size / (time.time() - start_batch_time)))
        
        mx.nd.waitall()
    
        print('[Epoch %d] throughput %.2f samples/s'%(
                    epoch, (args.batch_size * nbatch_train) / (time.time() - start_epoch_time)))
        

        val_L = eval(val_data)

        print('[Epoch %d] time cost %.2fs, valid loss %.2f, valid ppl %.2f'%(
            epoch, time.time()-start_epoch_time, val_L, math.exp(val_L)))

        if val_L < best_val:
            print("start val_L<best_val")
            best_val = val_L
            print("start test_L")
            test_L = eval(test_data)
            print("end test_L")
            model.collect_params().save(args.save)
            print('test loss %.2f, test ppl %.2f'%(test_L, math.exp(test_L)))
#             TODO: remove by referring to the paper, but may be useful
#         else:
#             print("start args.lr = args.lr*0.25")
#             args.lr = args.lr*0.25
#             trainer._init_optimizer('sgd',
#                                     {'learning_rate': args.lr,
#                                      'momentum': 0,
#                                      'wd': 0})
#             print("end trainer._init_optimizer")
#             model.collect_params().load(args.save, context)
#             print("end model.collect_params()")
            
    print('Total training throughput %.2f samples/s'%(
                            (args.batch_size * nbatch_train * args.epochs) / (time.time() - start_train_time)))
            

            
# TODO: add multi-GPU training support

# def train(num_gpus, batch_size, lr):
#     pass

if __name__ == '__main__':
    start_pipeline_time = time.time()
    if not args.eval_only:
        train()
    model.collect_params().load(args.save, context)
    val_L = eval(val_data)
    test_L = eval(test_data)
    print('Best validation loss %.2f, test ppl %.2f'%(val_L, math.exp(val_L)))
    print('Best test loss %.2f, test ppl %.2f'%(test_L, math.exp(test_L)))
    print('Total time cost %.2fs'%(time.time()-start_pipeline_time))
