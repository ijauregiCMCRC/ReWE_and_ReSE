# Supervised Neural Machine Translation - ReWE and ReSE

This code has been developed building upon the open sourced OpenNMT-py toolkit.

Table of Contents
=================
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
 
## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

Note that this code was only tested with PyTorch 0.4.

## Quickstart


### Step 1: Preprocess the data

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. 


After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

```bash
python train.py -data data/demo -save_model demo-model
```

The main train command is quite simple. Minimally it takes a data file
and a save file.  This will run the default model, which consists of a
2-layer LSTM with 500 hidden units on both the encoder/decoder. You
can also add `-gpuid 1` to use (say) GPU 1.

If you want to reproduce the results from the paper using ReWE and ReSE together, you need to add the following aguments to the above command:

```bash
python train.py -data data-path -save_model save-path -encoder_type brnn -rnn_type LSTM -rnn_size 1024 -word_vec_size 300 -global_attention mlp -optim adam  -learning_rate 0.0002 -neubig_style_training True -pre_word_vecs_enc pre-trained-file-enc -pre_word_vecs_dec pre-trained-file-dec -gpuid 0 -seed 1 -ReWE True -lamda_ReWE_loss 20 -ReSE True -lambda_ReSE_loss 100
```

The command above trains the model in the same way proposed by [(Denkowski and Neubig, 2018)](https://arxiv.org/abs/1706.09733), with simulated annealing and training until perplexity convergence.

Finally, there is another option (-contrastive_B) to train the systen using only ReWE in the loss function, and ignoring the negative log-likelihood (NLL) loss (see paper Appendix C). 

### Step 3: Translate

```bash
python translate.py -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.

If you want to predict the sentences using the regressed word embeddings and a nearest neighbour search, you can use the option '-emb_decoding'.

### Data

Data used in the paper can be downloaded from the [IWSLT 2016](https://sites.google.com/site/iwsltevaluation2016/data-provided) (for en-fr and cs-en) and from [WMT16 IT-domain translation task](http://www.statmt.org/wmt16/it-translation-task.html) (eu-en).

Fasttext embeddings can be downloaded from [here](https://fasttext.cc/docs/en/crawl-vectors.html).
