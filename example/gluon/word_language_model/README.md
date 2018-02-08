# Word-level language modeling RNN


## Usage

Example runs and the results:

```
python train.py --model lstm --gpus 1 --tied --nhid 256 --emsize 256 --nlayers 2 --lr 1.0 --clip 0.2 --epochs 40 --batch_size 32 --bptt 35 --dropout 0.2 --weight_drop 0.5 --save model.params          # Test ppl of 111.70 in wikitext-2 (example results), Test ppl of 66.48 in wikitext-2 (best results, still need to be finalized)
```


<br>

`python train.py --help` gives the following arguments:
```
Optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --dropout_h DROPOUT_H
                        dropout applied to hidden layer (0 = no dropout)
  --dropout_i DROPOUT_I
                        dropout applied to input layer (0 = no dropout)
  --dropout_e DROPOUT_E
                        dropout applied to embedding layer (0 = no dropout)
  --weight_dropout WEIGHT_DROPOUT
                        weight dropout applied to h2h weight matrix (0 = no
                        weight dropout)
  --tied                tie the word embedding and softmax weights
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --gctype GCTYPE       type of gradient compression to use, takes `2bit` or
                        `none` for now.
  --gcthreshold GCTHRESHOLD
                        threshold for 2bit gradient compression
  --eval_only           Whether to only evaluate the trained model
  --gpus GPUS           list of gpus to run, e.g. 0 or 0,2,5. empty means
                        using cpu (the result of multi-gpu training might be slightly different compared to single-gpu training, still need to be finalized)
```
